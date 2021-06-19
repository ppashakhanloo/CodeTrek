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
    #label_dict = load_label_dict(os.path.join(cmd_args.data_dir, cmd_args.data))
    node_types = {}
    edge_types = {}
    token_vocab = {}

    for key in [TOK_PAD, UNK]:
        get_or_add(node_types, key)
        get_or_add(edge_types, key)
        get_or_add(token_vocab, key)

    parse_util = WalkUtils if cmd_args.language == 'python' else JavaWalkUtils
    gh = GraphHolder()
    graph = RandomWalker.load_graph_from_gv(cmd_args.single_source)
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

    #tmp = '_'.join(cmd_args.single_source.split('_')[1:])
    #json_name = 'stub_' + '.'.join(tmp.split('.')[:-1]) + '.json'
    json_file = cmd_args.single_source.replace('graph_','stub_').replace('.gv', '.json')

    with open(json_file, 'r') as f:
        meta_data_list = json.load(f)
    for meta_data in meta_data_list:
        gh.add_graph(graph, meta_data)
        gh.dump('cooked_test')

