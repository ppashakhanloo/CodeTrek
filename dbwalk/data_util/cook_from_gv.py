import sys
import os
import json
import pickle as cp
from tqdm import tqdm
import numpy as np
import random
from data_prep.random_walk.walkutils import WalkUtils, JavaWalkUtils
from data_prep.random_walk.randomwalk import RandomWalker
from dbwalk.data_util.graph_holder import GraphHolder
from dbwalk.common.configs import cmd_args
from dbwalk.common.consts import TOK_PAD, var_idx2name, UNK
from dbwalk.data_util.cook_data import load_label_dict, get_or_add
from data_prep.tokenizer import tokenizer
import argparse


if __name__ == '__main__':
    label_dict = load_label_dict(os.path.join(cmd_args.data_dir, cmd_args.data))
    print(label_dict)
    node_types = {}
    edge_types = {}
    token_vocab = {}

    for key in [TOK_PAD, UNK]:
        get_or_add(node_types, key)
        get_or_add(edge_types, key)
        get_or_add(token_vocab, key)

    parse_util = WalkUtils if cmd_args.language == 'python' else JavaWalkUtils
    max_num_vars = 0
    print('building dict')
    for phase in ['train', 'dev', 'eval']:
        folder = os.path.join(cmd_args.data_dir, cmd_args.data, phase)
        files = os.listdir(folder)
        files = [fname for fname in files if fname.endswith('gv')]
        out_folder = os.path.join(cmd_args.data_dir, cmd_args.data, 'cooked_' + phase)
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)

        chunk_idx = 0
        gh = GraphHolder()
        pbar = tqdm(files)
        for fname in pbar:
            graph = RandomWalker.load_graph_from_gv(os.path.join(folder, fname))
            sep = '-' if '-' in fname else '_'
            json_name = sep.join(fname.split(sep)[1:])
            json_name = 'walks' + sep + '.'.join(json_name.split('.')[:-1]) + '.json'
            var_set = set()
            for node in graph.nodes(data=True):
                node_label = node[1]['label']
                node_name, values = parse_util.parse_node_label(node_label)
                node_type, node_value = parse_util.gen_node_type_value(node_name, values)
                if node_type.startswith('v_'):
                    var_set.add(node_type)
                else:
                    get_or_add(node_types, node_type)
                if len(node_value) != 0:
                    tok = tokenizer.tokenize(node_value, cmd_args.language)
                else:
                    tok = []
                node[1]['val_idx'] = [get_or_add(token_vocab, key) for key in tok]
                node[1]['raw_val'] = node_value
            for e in graph.edges(data=True):
                edge = e[2]['label']
                if edge[0] != '(':
                    edge = '(' + edge + ')'
                    e[2]['label'] = edge
                get_or_add(edge_types, edge)
            if len(var_set) > max_num_vars:
                max_num_vars = len(var_set)
            with open(os.path.join(folder, json_name), 'r') as f:
                meta_data = json.load(f)[0]
            gh.add_graph(graph, meta_data)
            if len(gh) >= cmd_args.data_chunk_size:
                gh.dump(os.path.join(out_folder, 'chunk_%d' % chunk_idx))
                chunk_idx += 1
                gh = GraphHolder()
            pbar.set_description('#n: %d, #e: %d, #v: %d, #t: %d' % (len(node_types), len(edge_types), max_num_vars, len(token_vocab)))
        if len(gh):
            gh.dump(os.path.join(out_folder, 'chunk_%d' % chunk_idx))
            
    print('# node types', len(node_types))
    print('# edge types', len(edge_types))
    print('max # vars per program', max_num_vars)
    print('# tokens', len(token_vocab))

    var_dict = {}
    var_reverse_dict = {}
    max_num_vars = min(max_num_vars, 100)
    for i in range(max_num_vars):
        val = get_or_add(node_types, var_idx2name(i))
        var_dict[i] = val
        var_reverse_dict[val] = i

    with open(os.path.join(cmd_args.data_dir, cmd_args.data, 'dict.pkl'), 'wb') as f:
        d = {}
        d['node_types'] = node_types
        d['edge_types'] = edge_types
        d['n_vars'] = max_num_vars
        d['var_dict'] = var_dict
        d['var_reverse_dict'] = var_reverse_dict
        d['token_vocab'] = token_vocab
        cp.dump(d, f, cp.HIGHEST_PROTOCOL)
