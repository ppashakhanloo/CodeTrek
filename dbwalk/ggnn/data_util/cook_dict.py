import json
import os
from tqdm import tqdm
import pickle as cp
from dbwalk.common.configs import cmd_args
from dbwalk.common.consts import TOK_PAD, var_idx2name, UNK

from dbwalk.data_util.cook_data import get_or_add
from data_prep.tokenizer import tokenizer


if __name__ == '__main__':
    node_types = {}
    edge_types = {}

    for key in [UNK]:
        get_or_add(node_types, key)
        get_or_add(edge_types, key)

    print('building dict')
    for phase in ['train', 'dev', 'eval']:
        folder = os.path.join(cmd_args.data_dir, cmd_args.data, phase)
        files = os.listdir(folder)
        for fname in tqdm(files):
            if not fname.endswith('json'):
                continue
            with open(os.path.join(folder, fname), 'r') as f:
                d = json.load(f)
            for k, v in d['ContextGraph']['NodeTypes'].items():
                get_or_add(node_types, v)
            for edge_type in d['ContextGraph']['Edges']:
                get_or_add(edge_types, edge_type)
                get_or_add(edge_types, 'inv-' + edge_type)
    var_dict = {}
    var_reverse_dict = {}
    print('# node types', len(node_types))
    print('# edge types', len(edge_types))

    with open(os.path.join(cmd_args.data_dir, cmd_args.data, 'dict.pkl'), 'wb') as f:
        d = {}
        d['node_types'] = node_types
        d['edge_types'] = edge_types
        d['n_vars'] = 0
        d['var_dict'] = var_dict
        d['var_reverse_dict'] = var_reverse_dict
        cp.dump(d, f, cp.HIGHEST_PROTOCOL)
