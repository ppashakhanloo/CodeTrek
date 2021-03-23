from __future__ import print_function, division

import os
import torch
import numpy as np
import pickle as cp
import random
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from dbwalk.common.consts import TOK_PAD, UNK
from dbwalk.rand_walk.walkutils import WalkUtils, JavaWalkUtils
from dbwalk.rand_walk.randomwalk import RandomWalker
from dbwalk.data_util.cook_data import make_mat_from_raw, get_or_unk
from dbwalk.data_util.graph_holder import MergedGraphHolders
from dbwalk.data_util.util import RawData, RawFile
from dbwalk.tokenizer import tokenizer


class ProgDict(object):
    def __init__(self, data_dir):
        with open(os.path.join(data_dir, 'dict.pkl'), 'rb') as f:
            d = cp.load(f)
        f_label = os.path.join(data_dir, 'all_labels.txt')
        self.label_map = {}
        with open(f_label, 'r') as f:
            for i, row in enumerate(f):
                row = row.strip()
                assert row not in self.label_map, 'duplicated label %s' % row
                self.label_map[row] = i
        self.node_types = d['node_types']
        self.edge_types = d['edge_types']
        self.node_val_tokens = d['token_vocab']
        self.max_num_vars = d['n_vars']

        self.var_dict = d['var_dict']
        self.var_reverse_dict = d['var_reverse_dict']
        print('# class', self.num_class)
        print('# node types', self.num_node_types)
        print('# edge types', self.num_edge_types)
        print('max # vars per program', self.max_num_vars)

    @property
    def num_node_val_tokens(self):
        return len(self.node_val_tokens)

    @property
    def num_class(self):
        return len(self.label_map)

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


def collate_raw_data(list_samples):
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

    full_node_idx = np.zeros((max_node_len, min_walks, len(list_samples)), dtype=np.int16)
    full_edge_idx = np.zeros((max_edge_len, min_walks, len(list_samples)), dtype=np.int16)

    for i, s in enumerate(list_samples):
        node_mat, edge_mat = s.node_idx, s.edge_idx
        full_node_idx[:node_mat.shape[0], :, i] = node_mat
        full_edge_idx[:edge_mat.shape[0], :, i] = edge_mat
    
    full_node_idx = torch.LongTensor(full_node_idx)
    full_edge_idx = torch.LongTensor(full_edge_idx)
    label = torch.LongTensor(label)

    if list_samples[0].node_val_idx is None:
        return full_node_idx, full_edge_idx, None, label
    else:
        _, word_dim = list_samples[0].node_val_idx
        sp_shape = (max_node_len, min_walks, len(list_samples), word_dim)
        list_coos = []
        word_dim = 0
        for i, s in enumerate(list_samples):
            coo, word_dim = s.node_val_idx
            row_ids = (coo[:, 0] * sp_shape[1] + coo[:, 1]) * sp_shape[2] + i
            list_coos.append(np.stack((row_ids, coo[:, 2])))
        list_coos = np.concatenate(list_coos, axis=1)
        node_val_mat = torch.sparse_coo_tensor(torch.LongTensor(list_coos), torch.ones((list_coos.shape[1],)),
                                               (sp_shape[0] * sp_shape[1] * sp_shape[2], sp_shape[3]))
        return full_node_idx, full_edge_idx, node_val_mat, label


class InMemDataest(Dataset):
    def __init__(self, args, prog_dict, data_dir, phase, sample_prob=None, shuffle_var=False):
        super(InMemDataest, self).__init__()

        self.prog_dict = prog_dict
        self.phase = phase
        self.sample_prob = sample_prob
        self.shuffle_var = shuffle_var
        assert self.prog_dict.node_idx(TOK_PAD) == self.prog_dict.edge_idx(TOK_PAD) == 0

        chunks = os.listdir(os.path.join(data_dir, 'cooked_' + phase))
        chunks = sorted(chunks)

        self.list_samples = []
        self.labeled_samples = defaultdict(list)
        for fname in chunks:
            with open(os.path.join(data_dir, 'cooked_' + phase, fname), 'rb') as f:
                d = cp.load(f)
                for key in d:
                    node_mat, edge_mat, src, str_label = d[key]
                    raw_sample = RawData(node_mat, edge_mat, None, src, self.prog_dict.label_map[str_label])
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
        if self.shuffle_var:
            var_remap = list(range(self.prog_dict.max_num_vars))
            random.shuffle(var_remap)
            for i in range(raw_sample.node_idx.shape[0]):
                for j in range(raw_sample.node_idx.shape[1]):
                    if raw_sample.node_idx[i, j] in self.prog_dict.var_reverse_dict:
                        old_var_idx = self.prog_dict.var_reverse_dict[raw_sample.node_idx[i, j]]
                        raw_sample.node_idx[i, j] = self.prog_dict.var_dict[var_remap[old_var_idx]]
        return raw_sample

    def get_test_loader(self, cmd_args):
        return DataLoader(self, batch_size=cmd_args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_raw_data, num_workers=0)

    def get_train_loader(self, cmd_args):
        return DataLoader(self, batch_size=cmd_args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_raw_data, num_workers=0)        


class WorkerContext(object):
    def __init__(self, worker_idx, tot_workers, total_num_samples):
        self.worker_idx = worker_idx
        self.tot_workers = tot_workers
        self.total_num_samples = total_num_samples
        assert worker_idx < self.tot_workers
        n_jobs = total_num_samples // self.tot_workers
        if total_num_samples % tot_workers:
            n_jobs += 1
        self.buffer = [None] * n_jobs


class AbstractOnlineWalkDB(Dataset):
    def __init__(self, args, prog_dict, data_dir, phase, sample_prob=None, shuffle_var=False):
        super(AbstractOnlineWalkDB, self).__init__()
        self.args = args
        self.prog_dict = prog_dict
        self.phase = phase
        self.sample_prob = sample_prob
        self.shuffle_var = shuffle_var
        assert self.prog_dict.node_idx(TOK_PAD) == self.prog_dict.edge_idx(TOK_PAD) == 0

    def get_train_loader(self, cmd_args):
        loader = DataLoader(self,
                            batch_size=cmd_args.batch_size,
                            shuffle=True,
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

    def get_item_from_rawfile(self, raw_sample, walker):
        walks = walker.random_walk(max_num_walks=self.args.num_walks, min_num_steps=self.args.min_steps, max_num_steps=self.args.max_steps)
        trajectories = [WalkUtils.build_trajectory(walk).to_dict() for walk in walks]
        node_mat, edge_mat = make_mat_from_raw(trajectories, self.prog_dict.node_types, self.prog_dict.edge_types)
        if self.args.use_node_val:
            node_val_coo = []
            for traj_idx, traj in enumerate(trajectories):
                for node_pos, node_val in enumerate(traj['node_values']):
                    toks = tokenizer.tokenize(node_val, self.args.language)
                    for tok in toks:
                        t = get_or_unk(self.prog_dict.node_val_tokens, tok)
                        node_val_coo.append((node_pos, traj_idx, t))
            node_val_coo = np.array(node_val_coo, dtype=np.int32)
        else:
            node_val_coo = None
        return RawData(node_mat, edge_mat, (node_val_coo, self.prog_dict.num_node_val_tokens), raw_sample.source, self.prog_dict.label_map[raw_sample.label])


class FastOnlineWalkDataset(AbstractOnlineWalkDB):
    def __init__(self, args, prog_dict, data_dir, phase, sample_prob=None, shuffle_var=False):
        super(FastOnlineWalkDataset, self).__init__(args, prog_dict, data_dir, phase, sample_prob, shuffle_var)

        chunks = os.listdir(os.path.join(data_dir, 'cooked_' + phase))
        chunks = sorted(chunks)
        chunks = [os.path.join(data_dir, 'cooked_' + phase, x) for x in chunks]
        self.merged_gh = MergedGraphHolders(chunks)
        self.language = 'python'

    def __len__(self):
        return len(self.merged_gh)

    def __getitem__(self, idx):
        raw_sample = self.merged_gh[idx]
        walker = RandomWalker(raw_sample.gv_file, raw_sample.anchor, self.language)
        return self.get_item_from_rawfile(raw_sample, walker)


class SlowOnlineWalkDataset(AbstractOnlineWalkDB):
    def __init__(self, args, prog_dict, data_dir, phase, sample_prob=None, shuffle_var=False):
        super(SlowOnlineWalkDataset, self).__init__(args, prog_dict, data_dir, phase, sample_prob, shuffle_var)

        gv_files = os.listdir(os.path.join(data_dir, phase))
        gv_files = [x for x in gv_files if x.endswith('.gv')]
        random.shuffle(gv_files)
        self.list_samples = []
        self.labeled_samples = defaultdict(list)
        for fname in gv_files:
            json_name = '_'.join(fname.split('_')[1:])
            json_name = 'walks_' + '.'.join(json_name.split('.')[:-1]) + '.json'
            with open(os.path.join(data_dir, phase, json_name), 'r') as f:
                data = json.load(f)[0]
                anchor_str = data['anchor']
                label = data['label']
                src = data['source']
                raw_sample = RawFile(os.path.join(data_dir, phase, fname), anchor_str, src, label)

            self.list_samples.append(raw_sample)
            self.labeled_samples[raw_sample.label].append(raw_sample)
        self.worker_context = WorkerContext(0, 1, len(self.list_samples))  # single proc loading for now
        language = 'python'
        print('loading graphs for', self.phase)
        for i, sample in tqdm(enumerate(self.list_samples)):
            graph = RandomWalker.load_graph_from_gv(sample.gv_file)
            walker = RandomWalker(graph, sample.anchor, language)
            self.worker_context.buffer[i] = walker

    def __len__(self):
        return len(self.list_samples)

    def __getitem__(self, idx):
        walker = self.worker_context.buffer[idx]
        assert walker is not None
        raw_sample = self.list_samples[idx]
        return self.get_item_from_rawfile(raw_sample, walker)
