import sys
import os
import json
import pickle as cp
import numpy as np
import networkx as nx
from collections import defaultdict
from dbwalk.data_util.util import RawFile

class GraphHolder(object):

    def __init__(self, is_directed=False):
        self.is_directed = is_directed
        self.num_graphs = 0
        self.tot_num_nodes = 0
        self.tot_num_edges = 0

        self.list_node_offset = []
        self.list_edge_offset = []        
        self.list_num_nodes = []
        self.list_num_edges = []
        self.edge_list = []

        self.list_anchors = []
        self.list_labels = []
        self.list_source = []
        self.list_node_labels = []
        self.list_node_tokens = []
        self.list_node_values = []
        self.edge_dict = {}
        self.inv_edge_dict = {}

    def _get_or_add_etype(self, e_type):
        if e_type in self.edge_dict:
            return self.edge_dict[e_type]
        val = len(self.edge_dict)
        self.edge_dict[e_type] = val
        self.inv_edge_dict[val] = e_type
        return val

    def add_graph(self, g, meta_info, node_index_base=1):
        self.num_graphs += 1

        self.list_node_offset.append(self.tot_num_nodes)
        self.list_edge_offset.append(self.tot_num_edges)        

        anchor_idx = None
        anchor_str = meta_info['anchor'] if 'anchor' in meta_info else None
        for idx, node in enumerate(g.nodes(data=True)):
            node_label = node[1]['label']
            node_idx = int(node[0]) - node_index_base
            assert idx == node_idx  # assume ordered
            self.list_node_labels.append(node_label)
            node_tok = ' '.join([str(x) for x in node[1]['val_idx']])
            raw_node_val = node[1]['raw_val']
            self.list_node_tokens.append(node_tok)
            self.list_node_values.append(raw_node_val)
            if node_label == anchor_str:
                anchor_idx = node_idx
        if anchor_idx is None:
            anchor_idx = meta_info['anchor_index']
            anchor_str = str(anchor_idx)
        self.list_anchors.append(anchor_str)
        self.list_labels.append(meta_info['label'])
        self.list_source.append(meta_info['source'])
        num_edges = 0
        for e in g.edges(data=True):
            edge = e[2]['label']
            e_type = self._get_or_add_etype(edge)
            x = int(e[0]) - node_index_base
            y = int(e[1]) - node_index_base
            self.edge_list.append((x, y, e_type))
            num_edges += 1

        self.list_num_nodes.append(len(g))
        self.list_num_edges.append(num_edges)
        self.tot_num_nodes += len(g)
        self.tot_num_edges += num_edges

    def _dump_int_arr(self, out_name, list_int):
        arr = np.array(list_int, dtype=np.int32)
        np.save(out_name, arr)

    def dump(self, out_folder):
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)
        for key in ['list_node_offset', 'list_edge_offset', 'list_num_nodes', 'list_num_edges', 'edge_list']:
            self._dump_int_arr(os.path.join(out_folder, key + '.npy'), getattr(self, key))
        for key in ['list_anchors', 'list_node_labels', 'list_node_tokens', 'list_node_values', 'list_labels', 'list_source']:
            with open(os.path.join(out_folder, key + '.txt'), 'w') as f:
                str_list = getattr(self, key)
                for row in str_list:
                    f.write('%s\n' % row)
        with open(os.path.join(out_folder, 'edge_dict.pkl'), 'wb') as f:
            cp.dump(self.edge_dict, f, cp.HIGHEST_PROTOCOL)

    def _load_int_arr(self, out_name, key):
        list_int = np.load(out_name).tolist()
        setattr(self, key, list_int)

    def load(self, out_folder):
        print('loading graphs from', out_folder)
        for key in ['list_node_offset', 'list_edge_offset', 'list_num_nodes', 'list_num_edges', 'edge_list']:
            self._load_int_arr(os.path.join(out_folder, key + '.npy'), key)

        for key in ['list_anchors', 'list_node_labels', 'list_node_tokens', 'list_node_values', 'list_labels', 'list_source']:
            str_list = []
            with open(os.path.join(out_folder, key + '.txt'), 'r') as f:
                for row in f:
                    str_list.append(row.rstrip())
            setattr(self, key, str_list)
        with open(os.path.join(out_folder, 'edge_dict.pkl'), 'rb') as f:
            self.edge_dict = cp.load(f)
            self.inv_edge_dict = {}
            for key in self.edge_dict:
                self.inv_edge_dict[self.edge_dict[key]] = key
        self.num_graphs = len(self.list_num_nodes)
        self.tot_num_nodes = sum(self.list_num_nodes)
        self.tot_num_edges = sum(self.list_num_edges)
        assert self.tot_num_nodes == len(self.list_node_labels) == len(self.list_node_tokens) == len(self.list_node_values)
        assert self.tot_num_edges == len(self.edge_list)
        assert self.num_graphs == len(self.list_node_offset) == len(self.list_edge_offset) == len(self.list_labels) 
        assert self.num_graphs == len(self.list_num_edges) == len(self.list_anchors) == len(self.list_source)
        print('%d graphs loaded' % self.num_graphs)

    def __getitem__(self, g_idx):
        assert g_idx >= 0 and g_idx < self.num_graphs
        if self.is_directed:
            g = nx.empty_graph(0, nx.MultiDiGraph)
        else:
            g = nx.empty_graph(0, nx.MultiGraph)

        node_offset = self.list_node_offset[g_idx]
        edge_offset = self.list_edge_offset[g_idx]

        for node_idx in range(self.list_num_nodes[g_idx]):
            g.add_node(node_idx, 
                       label=self.list_node_labels[node_offset + node_idx],
                       val_idx=[int(x) for x in self.list_node_tokens[node_offset + node_idx].split()],
                       raw_val=self.list_node_values[node_offset + node_idx])

        for e_idx in range(edge_offset, edge_offset + self.list_num_edges[g_idx]):
            u, v, etype_idx = self.edge_list[e_idx]
            etype = self.inv_edge_dict[etype_idx]
            g.add_edge(u, v, label=etype)
        sample = RawFile(g, self.list_anchors[g_idx], self.list_source[g_idx], self.list_labels[g_idx])
        return sample

    def __len__(self):
        return self.num_graphs


class MergedGraphHolders(object):
    def __init__(self, list_dumps, is_directed=False, sample_prob=None):
        self.list_gh = []
        self.num_graphs = 0
        self.is_directed = is_directed
        if sample_prob is not None:
            self.label_keys = list(sample_prob.keys())
            self.sample_prob = []
            for key in self.label_keys:
                self.sample_prob.append(sample_prob[key])
        else:
            self.sample_prob = self.label_keys = None
        self.labeled_samples = defaultdict(list)
        for dump_folder in list_dumps:
            gh = GraphHolder(is_directed)
            gh.load(dump_folder)
            if sample_prob is not None:
                for i in range(len(gh)):
                    self.labeled_samples[gh.list_labels[i]].append(self.num_graphs + i)
            self.num_graphs += len(gh)
            self.list_gh.append(gh)

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, g_idx):
        if self.sample_prob is not None:
            sample_cls = np.argmax(np.random.multinomial(1, self.sample_prob))
            samples = self.labeled_samples[self.label_keys[sample_cls]]
            idx = np.random.randint(len(samples))
            g_idx = samples[idx]
        assert g_idx >= 0 and g_idx < self.num_graphs
        prefix_sum = 0
        for gh in self.list_gh:
            if g_idx < prefix_sum + len(gh):
                return gh[g_idx - prefix_sum]
            prefix_sum += len(gh)
        assert False
