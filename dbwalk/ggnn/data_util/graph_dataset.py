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

class ProgGraph(object):
    def __init__(self, d, prog_dict):
        graph = d['ContextGraph']
        node_labels = graph['NodeLabels']

        self.label = prog_dict.label_map[d['label']]
        self.target_idx = d['SlottedNodeIdx'] - 1
        self.num_node_feats = len(prog_dict.node_types)
        self.num_nodes = len(node_labels)
        self.typed_edge_list = [[] for _ in range(len(prog_dict.edge_types))]
        self.edge_list = []

        edge_dict = graph['Edges']
        for e_type in edge_dict:
            fwd_idx = prog_dict.edge_types[e_type]
            backwd_idx = prog_dict.edge_types['inv-' + e_type]
            for x, y in edge_dict[e_type]:
                x = x - 1
                y = y - 1
                self.typed_edge_list[fwd_idx].append((x, y))
                self.typed_edge_list[backwd_idx].append((y, x))
            self.edge_list.append((x, y, fwd_idx))
            self.edge_list.append((y, x, fwd_idx))
        self.node_list = [None] * self.num_nodes
        for node_idx in node_labels:
            self.node_list[int(node_idx) - 1] = prog_dict.node_types[node_labels[node_idx]]


def collate_graph_data(list_samples):
    labels = []
    for g in list_samples:
        labels.append(g.label)
    labels = torch.LongTensor(labels)
    return list_samples, labels


class AstGraphDataset(Dataset):
    def __init__(self, args, prog_dict, data_dir, phase, sample_prob=None):
        super(AstGraphDataset, self).__init__()
        self.prog_dict = prog_dict
        self.phase = phase
        self.sample_prob = sample_prob
        self.gnn_type = args.gnn_type

        json_files = os.listdir(os.path.join(data_dir, phase))
        json_files = [x for x in json_files if x.endswith('.json')]
        random.shuffle(json_files)
        self.list_samples = []
        self.labeled_samples = defaultdict(list)
        for json_name in json_files:
            with open(os.path.join(data_dir, phase, json_name), 'r') as f:
                d = json.load(f)
                self.list_samples.append(d)
                label = self.prog_dict.label_map[d['label']]
            self.labeled_samples[label].append(d)

    def __len__(self):
        return len(self.list_samples)

    def __getitem__(self, idx):
        if self.sample_prob is None:
            sample_dict = self.list_samples[idx]
        else:
            assert self.phase == 'train'
            sample_cls = np.argmax(np.random.multinomial(1, self.sample_prob))
            samples = self.labeled_samples[sample_cls]
            idx = np.random.randint(len(samples))
            sample_dict = samples[idx]
        pg = ProgGraph(sample_dict, self.prog_dict)
        return get_gnn_graph(pg, self.gnn_type)

    def get_train_loader(self, cmd_args):
        loader = DataLoader(self,
                            batch_size=cmd_args.batch_size,
                            shuffle=True,
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

    