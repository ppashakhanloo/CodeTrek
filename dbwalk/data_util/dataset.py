from __future__ import print_function, division

import os
import torch
import numpy as np
import pickle as cp

from torch.utils.data import Dataset, DataLoader
from collections import namedtuple, defaultdict
from dbwalk.common.consts import TOK_PAD, UNK

RawData = namedtuple('RawData', ['node_idx', 'edge_idx', 'source', 'label'])


class ProgDict(object):
    def __init__(self, data_dir):
        with open(os.path.join(data_dir, 'dict.pkl'), 'rb') as f:
            d = cp.load(f)
        self.node_types = d['node_types']
        self.edge_types = d['edge_types']
        self.max_num_vars = d['n_vars']
        print('# node types', len(self.node_types))
        print('# edge types', len(self.edge_types))
        print('max # vars per program', self.max_num_vars)

    @property
    def num_node_types(self):
        return len(self.node_types)
    
    @property
    def num_edge_types(self):
        return len(self.edge_types)

    def node_idx(self, node_name):
        if node_name in self.node_types:
            return self.node_types[node_name]
        return self.node_types[UNK]

    def edge_idx(self, edge_name):
        if edge_name in self.edge_types:
            return self.edge_types[edge_name]
        return self.edge_types[UNK]


class InMemDataest(Dataset):
    def __init__(self, prog_dict, data_dir, phase, sample_prob=None):
        super(InMemDataest, self).__init__()
                
        self.prog_dict = prog_dict
        self.phase = phase
        self.sample_prob = sample_prob
        assert self.prog_dict.node_idx(TOK_PAD) == self.prog_dict.edge_idx(TOK_PAD) == 0

        f_label = os.path.join(data_dir, 'all_labels.txt')
        self.label_map = {}
        with open(f_label, 'r') as f:
            for i, row in enumerate(f):
                row = row.strip()
                assert row not in self.label_map, 'duplicated label %s' % row
                self.label_map[row] = i
        chunks = os.listdir(os.path.join(data_dir, 'cooked_' + phase))
        chunks = sorted(chunks)

        self.list_samples = []
        self.labeled_samples = defaultdict(list)
        for fname in chunks:
            with open(os.path.join(data_dir, 'cooked_' + phase, fname), 'rb') as f:
                d = cp.load(f)
                for key in d:
                    node_mat, edge_mat, src, str_label = d[key]
                    raw_sample = RawData(node_mat, edge_mat, src, self.label_map[str_label])
                    self.list_samples.append((key, raw_sample))
                    self.labeled_samples[raw_sample.label].append((key, raw_sample))

        print('# samples in %s: %d' % (phase, len(self.list_samples)))

    def __len__(self):
        return len(self.list_samples)
    
    def __getitem__(self, idx):
        if self.sample_prob is None:
            _, raw_sample = self.list_samples[idx]
        else:
            assert self.phase == 'train'
            sample_cls = np.argmax(np.random.multinomial(1, self.sample_prob))
            samples = self.labeled_samples[sample_cls]
            idx = np.random.randint(len(samples))
            _, raw_sample = samples[idx]
        return raw_sample

    def collate_fn(self, list_samples):
        label = []
        min_walks = list_samples[0].node_idx.shape[1]
        max_walks = 0
        max_node_len = 0
        max_edge_len = 0
        for s in list_samples:
            label.append(s.label)
            max_node_len = max(s.node_idx.shape[0], max_node_len)
            max_edge_len = max(s.edge_idx.shape[0], max_edge_len)            
            min_walks = min(s.node_idx.shape[1], min_walks)
            max_walks = max(s.node_idx.shape[1], max_walks)

        if min_walks != max_walks:
            print('warning: right now only support fixed number of walks')
            print('giving up %d samples in a batch' % len(label))
            return None, None, None

        full_node_idx = np.zeros((max_node_len, len(list_samples), min_walks), dtype=np.int16)
        full_edge_idx = np.zeros((max_edge_len, len(list_samples), min_walks), dtype=np.int16)

        for i, s in enumerate(list_samples):
            node_mat, edge_mat = s.node_idx, s.edge_idx
            full_node_idx[:node_mat.shape[0], i, :] = node_mat
            full_edge_idx[:edge_mat.shape[0], i, :] = edge_mat
        
        full_node_idx = torch.LongTensor(full_node_idx)
        full_edge_idx = torch.LongTensor(full_edge_idx)
        label = torch.LongTensor(label)
        return full_node_idx, full_edge_idx, label
