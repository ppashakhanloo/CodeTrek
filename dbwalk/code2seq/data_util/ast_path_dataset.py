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
from dbwalk.data_util.util import RawData
from dbwalk.data_util.dataset import ProgDict
from dbwalk.data_util.graph_holder import MergedGraphHolders


class AstTree(object):
    def __init__(self, sample, prog_dict):
        graph = sample.gv_file
        self.graph = graph
        self.label = prog_dict.label_map[sample.label]
        self.num_nodes = len(graph)
        self.root = 1
        self.parent_list = [-1] * self.num_nodes
        self.is_leaf = [True] * self.num_nodes
        for e in graph.edges():
            x, y = e
            self.parent_list[y] = x
            self.is_leaf[x] = False
        self.target_idxs = []
        for i in sample.anchors[1:-1].split(', '):
            self.target_idxs.append(int(i[1:-1]))
        self.leaves = [x for x in range(self.num_nodes) if self.is_leaf[x]]

        self.node_type_list = [None] * self.num_nodes
        self.node_val_list = [None] * self.num_nodes
        for idx, node in enumerate(graph.nodes(data=True)):
            self.node_type_list[idx] = prog_dict.node_types[node[1]['label']]
            self.node_val_list[idx] = node[1]['val_idx']

    def _get_path2root(self, x):
        p = []
        while x != self.root:            
            x = self.parent_list[x]
            p.append(x)
        return p

    def sample_paths(self, num_paths, max_steps):
        node_val_coo = []
        list_traj = []
        max_len = 0
        for traj_idx in range(num_paths):
            x, y = np.random.choice(self.leaves, 2, replace=False)
            p1 = self._get_path2root(x)
            p2 = self._get_path2root(y)
            path = p1[:-1] + p2[::-1]
            if len(path) > max_steps:
                st = np.random.randint(len(path) - max_steps + 1)
                path = path[st:st+max_steps]
            max_len = max(max_len, len(path))
            for i in self.node_val_list[x]:
                node_val_coo.append((0, traj_idx, i))
            for i in self.node_val_list[y]:
                node_val_coo.append((1, traj_idx, i))
            path = [self.node_type_list[i] for i in path]
            list_traj.append(path)
        mat_path = np.zeros((max_len, num_paths), dtype=np.int16)
        for i, path in enumerate(list_traj):
            mat_path[:len(path), i] = path
        return mat_path, np.array(node_val_coo, dtype=np.int32)


def collate_raw_data(list_samples):
    label = []
    max_node_len = 0
    num_paths = list_samples[0].node_idx.shape[1]
    for s in list_samples:
        label.append(s.label)
        max_node_len = max(s.node_idx.shape[0], max_node_len)

    full_node_idx = np.zeros((max_node_len, num_paths, len(list_samples)), dtype=np.int16)
    for i, s in enumerate(list_samples):
        node_mat = s.node_idx
        full_node_idx[:node_mat.shape[0], :, i] = node_mat
    full_node_idx = torch.LongTensor(full_node_idx)
    label = torch.LongTensor(label)

    _, word_dim = list_samples[0].node_val_idx
    sp_shape = (2, num_paths, len(list_samples), word_dim)
    list_coos = []
    for i, s in enumerate(list_samples):
        coo, word_dim = s.node_val_idx
        if coo.shape[0]:
            row_ids = (coo[:, 0] * sp_shape[1] + coo[:, 1]) * sp_shape[2] + i
            list_coos.append(np.stack((row_ids, coo[:, 2])))
    list_coos = np.concatenate(list_coos, axis=1)
    node_val_mat = (torch.LongTensor(list_coos), torch.ones((list_coos.shape[1],)),
                    (sp_shape[0] * sp_shape[1] * sp_shape[2], sp_shape[3]))
    return full_node_idx, None, node_val_mat, label


class AstPathDataset(Dataset):
    def __init__(self, args, prog_dict, data_dir, phase, sample_prob=None):
        super(AstPathDataset, self).__init__()
        self.prog_dict = prog_dict
        self.args = args
        self.phase = phase
        self.max_steps = args.max_steps
        if self.phase != 'train':
            assert sample_prob is None

        chunks = os.listdir(os.path.join(data_dir, 'cooked_' + phase))
        chunks = sorted(chunks)
        chunks = [os.path.join(data_dir, 'cooked_' + phase, x) for x in chunks]
        self.merged_gh = MergedGraphHolders(chunks, is_directed=True, sample_prob=sample_prob)

    def __len__(self):
        return len(self.merged_gh)

    def __getitem__(self, idx):
        raw_sample = self.merged_gh[idx]
        tree = AstTree(raw_sample, self.prog_dict)
        mat_path, node_val_coo = tree.sample_paths(self.args.num_walks, self.max_steps)
        return RawData(mat_path, None, (node_val_coo, self.prog_dict.num_node_val_tokens), raw_sample.source, self.prog_dict.label_map[raw_sample.label])

    def get_train_loader(self, cmd_args):
        loader = DataLoader(self,
                            batch_size=cmd_args.batch_size,
                            shuffle=False,
                            drop_last=True,
                            collate_fn=collate_raw_data,
                            num_workers=cmd_args.num_proc)
        return loader

    def get_test_loader(self, cmd_args):
        loader = DataLoader(self,
                            batch_size=cmd_args.batch_size,
                            shuffle=False,
                            drop_last=False,
                            collate_fn=collate_raw_data,
                            num_workers=cmd_args.num_proc)
        return loader
