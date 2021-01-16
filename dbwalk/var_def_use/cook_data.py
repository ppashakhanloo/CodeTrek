import sys
import os
import json
import pickle as cp
from tqdm import tqdm
import numpy as np
import random
from dbwalk.common.configs import cmd_args
from dbwalk.common.consts import TOK_PAD


def get_or_add(type_dict, key):
    if key in type_dict:
        return type_dict[key]
    val = len(type_dict)
    type_dict[key] = val
    return val


def dump_data_chunk(out_folder, chunk_idx, chunk_buf):
    fname = os.path.join(out_folder, 'chunk_%d.pkl' % chunk_idx)
    with open(fname, 'wb') as f:
        cp.dump(chunk_buf, f, cp.HIGHEST_PROTOCOL)
    return chunk_idx + 1, {}


def make_mat(list_traj, max_n_nodes, max_n_edges):
    assert node_types[TOK_PAD] == 0 and edge_types[TOK_PAD] == 0  # zero padding
    mat_node = np.zeros((max_n_nodes, len(list_traj)), dtype=np.int16)
    mat_edge = np.zeros((max_n_edges, len(list_traj)), dtype=np.int16)

    for i, (seq_nodes, seq_edges) in enumerate(list_traj):
        mat_node[:len(seq_nodes), i] = seq_nodes
        mat_edge[:len(seq_edges), i] = seq_edges
    return mat_node, mat_edge


if __name__ == '__main__':

    node_types = {}
    edge_types = {}

    for key in [TOK_PAD]:
        get_or_add(node_types, key)
        get_or_add(edge_types, key)

    max_num_vars = 0
    print('building dict')
    for phase in ['train', 'dev', 'test']:
        folder = os.path.join(cmd_args.data_dir, cmd_args.data, phase)
        files = os.listdir(folder)
        for fname in tqdm(files):
            if not fname.endswith('json'):
                continue
            with open(os.path.join(folder, fname), 'r') as f:
                d = json.load(f)
                for sample in d:
                    var_set = set()
                    for traj in sample['trajectories']:
                        for node in traj['nodes']:
                            if node.startswith('v_'):
                                var_set.add(node)
                            else:
                                get_or_add(node_types, node)
                        for edge in traj['edges']:
                            get_or_add(edge_types, edge)
                    if len(var_set) > max_num_vars:
                        max_num_vars = len(var_set)
    print('# node types', len(node_types))
    print('# edge types', len(edge_types))
    print('max # vars per program', max_num_vars)

    for i in range(max_num_vars):
        get_or_add(node_types, 'var_%d' % i)

    with open(os.path.join(cmd_args.data_dir, cmd_args.data, 'dict.pkl'), 'wb') as f:
        d = {}
        d['node_types'] = node_types
        d['edge_types'] = edge_types
        d['n_vars'] = max_num_vars
        cp.dump(d, f, cp.HIGHEST_PROTOCOL)

    for phase in ['train', 'dev', 'test']:
        print('cooking', phase)
        folder = os.path.join(cmd_args.data_dir, cmd_args.data, phase)
        files = os.listdir(folder)
        random.shuffle(files)
        out_folder = os.path.join(cmd_args.data_dir, cmd_args.data, 'cooked_' + phase)
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)

        chunk_idx = 0
        chunk_buf = {}
        for fname in tqdm(files):
            if not fname.endswith('json'):
                continue
            fname_prefix = '.'.join(fname.split('.')[:-1])
            with open(os.path.join(folder, fname), 'r') as f:
                d = json.load(f)
                for sample_idx, sample in enumerate(d):
                    var_dict = {}
                    list_traj = []
                    max_len_nodes = 0
                    max_len_edges = 0
                    for traj in sample['trajectories']:
                        seq_nodes = []
                        max_len_nodes = max(max_len_nodes, len(traj['nodes']))
                        max_len_edges = max(max_len_edges, len(traj['edges']))
                        for node in traj['nodes']:
                            if node.startswith('v_'):
                                v_idx = get_or_add(var_dict, node)
                                seq_nodes.append(node_types['var_%d' % v_idx])
                            else:
                                seq_nodes.append(node_types[node])
                        seq_edges = [edge_types[e] for e in traj['edges']]
                        list_traj.append((seq_nodes, seq_edges))
                    node_mat, edge_mat = make_mat(list_traj, max_len_nodes, max_len_edges)
                    chunk_buf['%s-%d' % (fname_prefix, sample_idx)] = (node_mat, edge_mat, sample['source'], sample['label'])
            if len(chunk_buf) >= cmd_args.data_chunk_size:
                chunk_idx, chunk_buf = dump_data_chunk(out_folder, chunk_idx, chunk_buf) 
        if len(chunk_buf):
            dump_data_chunk(out_folder, chunk_idx, chunk_buf)
