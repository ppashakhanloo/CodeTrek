from __future__ import print_function, division

import os
import torch
import numpy as np
import json
import random
import pickle as cp
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple, defaultdict
from dbwalk.common.consts import TOK_PAD, UNK

from dbwalk.data_util.dataset import ProgDict
from dbwalk.ggnn.graphnet.s2v_lib import get_gnn_graph
from dbwalk.data_util.graph_holder import MergedGraphHolders


class ProgGraph(object):
    def __init__(self, sample, prog_dict):
        graph = sample.gv_file
        self.label = prog_dict.label_map[sample.label]
        self.node_val_dim = prog_dict.num_node_val_tokens
        if sample.anchors != 'None':
            self.target_idx = [int(anch[1:-1]) for anch in sample.anchors[1:-1].split(', ')]
        else:
            self.target_idx = None
        self.num_node_feats = len(prog_dict.node_types)
        self.num_nodes = len(graph)
        self.typed_edge_list = [[] for _ in range(len(prog_dict.edge_types))]
        self.edge_list = []


        for e in graph.edges(data=True):
            e_type = e[2]['label']
            fwd_idx = prog_dict.edge_types[e_type]
            backwd_idx = prog_dict.edge_types['inv-' + e_type]            
            x = int(e[0])
            y = int(e[1])
            self.edge_list.append((x, y, fwd_idx))
            self.edge_list.append((y, x, backwd_idx))
            self.typed_edge_list[fwd_idx].append((x, y))
            self.typed_edge_list[backwd_idx].append((y, x))
        self.node_list = [None] * self.num_nodes
        self.node_val_coo = []
        for idx, node in enumerate(graph.nodes(data=True)):
            assert idx == int(node[0])
            self.node_list[idx] = prog_dict.node_types[node[1]['label']]
            for i in node[1]['val_idx']:
                self.node_val_coo.append((idx, i))
        if len(self.node_val_coo):
            self.node_val_coo = np.array(self.node_val_coo, dtype=np.int32).T
        else:
            self.node_val_coo = None


def collate_graph_data(list_samples):
    labels = []
    list_val_coo = []
    offset = 0
    for g in list_samples:
        labels.append(g.label)
        if g.pg.node_val_coo is not None:
            cur_coo = g.pg.node_val_coo.copy()
            cur_coo[0, :] += offset
            list_val_coo.append(cur_coo)
        offset += g.num_nodes
    list_val_coo = np.concatenate(list_val_coo, axis=1)
    labels = torch.LongTensor(labels)
    node_val_mat = (torch.LongTensor(list_val_coo), torch.ones((list_val_coo.shape[1],)),
                    (offset, g.pg.node_val_dim))
    return list_samples, node_val_mat, labels


class AstGraphDataset(Dataset):
    def __init__(self, args, prog_dict, data_dir, phase, sample_prob=None):
        super(AstGraphDataset, self).__init__()
        self.prog_dict = prog_dict
        self.phase = phase
        self.sample_prob = sample_prob
        self.gnn_type = args.gnn_type

        chunks = os.listdir(os.path.join(data_dir, 'cooked_' + phase))
        chunks = sorted(chunks)
        chunks = [os.path.join(data_dir, 'cooked_' + phase, x) for x in chunks]
        self.merged_gh = MergedGraphHolders(chunks, is_directed=True, sample_prob=sample_prob)

    def __len__(self):
        return len(self.merged_gh)

    def __getitem__(self, idx):
        raw_sample = self.merged_gh[idx]
        pg = ProgGraph(raw_sample, self.prog_dict)
        return get_gnn_graph(pg, self.gnn_type)

    def get_train_loader(self, cmd_args):
        loader = DataLoader(self,
                            batch_size=cmd_args.batch_size,
                            shuffle=False,
                            drop_last=True,
                            collate_fn=collate_graph_data,
                            num_workers=cmd_args.num_proc)
        return loader

    def get_test_loader(self, cmd_args):
        loader = DataLoader(self,
                            batch_size=cmd_args.batch_size,
                            shuffle=False,
                            drop_last=False,
                            collate_fn=collate_graph_data,
                            num_workers=cmd_args.num_proc)
        return loader


if __name__ == '__main__':
    from dbwalk.common.configs import cmd_args, set_device
    set_device(cmd_args.gpu)
    prog_dict = ProgDict(cmd_args.data_dir)
    db = AstGraphDataset(cmd_args, prog_dict, cmd_args.data_dir, 'train')
    g = db[0]
