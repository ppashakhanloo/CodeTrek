import json
import os
from tqdm import tqdm
import networkx as nx
import pickle as cp
from dbwalk.common.configs import cmd_args
from dbwalk.common.consts import TOK_PAD, var_idx2name, UNK
from dbwalk.data_util.graph_holder import GraphHolder
from dbwalk.data_util.cook_data import get_or_add
from dbwalk.tokenizer import tokenizer


if __name__ == '__main__':
    node_types = {}
    edge_types = {}
    token_vocab = {}

    for key in [UNK]:
        get_or_add(node_types, key)
        get_or_add(edge_types, key)
        get_or_add(token_vocab, key)

    print('building dict')
    for phase in ['train', 'dev', 'eval']:
        out_folder = os.path.join(cmd_args.data_dir, cmd_args.data, 'cooked_' + phase)
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)
        
        folder = os.path.join(cmd_args.data_dir, cmd_args.data, phase)
        files = os.listdir(folder)
        chunk_idx = 0
        gh = GraphHolder()
        pbar = tqdm(files)
        for fname in pbar:
            if not fname.endswith('json'):
                continue
            with open(os.path.join(folder, fname), 'r') as f:
                d = json.load(f)
            g = nx.empty_graph(0, nx.MultiGraph)
            meta_info = {'anchor_index': d['SlotNodeIdx'],
                         'source': d['filename'],
                         'label': d['label']}
            node_vals = d['ContextGraph']['NodeValues']
            node_toks = d['ContextGraph']['NodeTokens']
            for i, v in enumerate(d['ContextGraph']['NodeTypes']):
                get_or_add(node_types, v)
                tok = [] if len(node_vals[i]) == 0 else node_toks[i]
                val_idx = [get_or_add(token_vocab, key) for key in tok]
                g.add_node(i, label=v, val_idx=val_idx, raw_val='')
            for edge_type in d['ContextGraph']['Edges']:
                get_or_add(edge_types, edge_type)
                for e in d['ContextGraph']['Edges'][edge_type]:
                    g.add_edge(e[0], e[1], label=edge_type)
            gh.add_graph(g, meta_info, node_index_base=0)
            if len(gh) >= cmd_args.data_chunk_size:
                gh.dump(os.path.join(out_folder, 'chunk_%d' % chunk_idx))
                chunk_idx += 1
                gh = GraphHolder()
            pbar.set_description('#n: %d, #e: %d, #t: %d' % (len(node_types), len(edge_types), len(token_vocab)))
        if len(gh):
            gh.dump(os.path.join(out_folder, 'chunk_%d' % chunk_idx))            
    var_dict = {}
    var_reverse_dict = {}
    fwd_types = list(edge_types.keys())
    for etype in fwd_types:
        get_or_add(edge_types, 'inv-' + etype)
    print('# node types', len(node_types))
    print('# edge types', len(edge_types))
    print('# tokens', len(token_vocab))

    with open(os.path.join(cmd_args.data_dir, cmd_args.data, 'dict.pkl'), 'wb') as f:
        d = {}
        d['node_types'] = node_types
        d['edge_types'] = edge_types
        d['n_vars'] = 0
        d['var_dict'] = var_dict
        d['var_reverse_dict'] = var_reverse_dict
        d['token_vocab'] = token_vocab
        cp.dump(d, f, cp.HIGHEST_PROTOCOL)
