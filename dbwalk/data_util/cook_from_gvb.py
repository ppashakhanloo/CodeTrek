import os
import json
import pickle as cp
from tqdm import tqdm
from data_prep.random_walk.walkutils import WalkUtils, JavaWalkUtils
from data_prep.random_walk.randomwalk import RandomWalker
from data_prep.graph.graphutils import GraphUtils

from dbwalk.common.configs import cmd_args
from dbwalk.common.consts import TOK_PAD, var_idx2name, UNK
from dbwalk.data_util.cook_data import load_label_dict, get_or_add


if __name__ == '__main__':
    label_dict = load_label_dict(os.path.join(cmd_args.data_dir, cmd_args.data))
    print(label_dict)
    node_types = {}
    edge_types = {}

    for key in [TOK_PAD, UNK]:
        get_or_add(node_types, key)
        get_or_add(edge_types, key)

    max_num_vars = 0
    print('building dict')
    for phase in ['train', 'dev', 'eval']:
        folder = os.path.join(cmd_args.data_dir, cmd_args.data, phase)
        files = os.listdir(folder)
        files = [fname for fname in files if fname.endswith('gvb')]
        for fname in tqdm(files):
            graph = GraphUtils.deserialize(os.path.join(folder, fname))
            json_name = '_'.join(fname.split('_')[1:])
            json_name = 'walks_' + '.'.join(json_name.split('.')[:-1]) + '.json'
            with open(os.path.join(folder, json_name), 'r') as f:
                data = json.load(f)[0]
                anchor_str = data['anchor']
            walker = RandomWalker(graph, anchor_str, cmd_args.language)
            walks = walker.random_walk(max_num_walks=cmd_args.num_walks, min_num_steps=cmd_args.min_steps, max_num_steps=cmd_args.max_steps)
            if cmd_args.language == 'python':
                trajectories = [WalkUtils.build_trajectory(walk) for walk in walks]
            elif cmd_args.language == 'java':
                trajectories = [JavaWalkUtils.build_trajectory(walk) for walk in walks]
            else:
                raise ValueError('Unknown language:', cmd_args.language)
            var_set = set()
            for trajectory in trajectories:
                traj = trajectory.to_dict()
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

    var_dict = {}
    var_reverse_dict = {}
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
        cp.dump(d, f, cp.HIGHEST_PROTOCOL)
