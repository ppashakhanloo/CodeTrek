from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import ctypes
import numpy as np
import os
import sys
import torch
import json
from torch.autograd import Variable
from dbwalk.common.consts import t_float


class _s2v_lib(object):

    def __init__(self, args):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.lib = ctypes.CDLL('%s/build/dll/libs2v.so' % dir_path)

        self.lib.n2n_construct.restype = ctypes.c_int
        self.lib.prepare_indices.restype = ctypes.c_int

        if sys.version_info[0] > 2:
            args = [arg.encode() for arg in args]  # str -> bytes for each element in args
        arr = (ctypes.c_char_p * len(args))()
        arr[:] = args
        self.lib.Init(len(args), arr)

    def _getGraphOrList(self, s2v_graphs):
        if type(s2v_graphs) is not list:
            single_graph = s2v_graphs
        elif len(s2v_graphs) == 1:
            single_graph = s2v_graphs[0]
        else:
            single_graph = None
        return single_graph

    def PrepareIndices(self, graph_list, device, fn_edges=lambda x: x.edge_pairs):
        edgepair_list = (ctypes.c_void_p * len(graph_list))()
        list_num_nodes = np.zeros((len(graph_list), ), dtype=np.int32)
        list_num_edges = np.zeros((len(graph_list), ), dtype=np.int32)        
        for i in range(len(graph_list)):
            edges = fn_edges(graph_list[i])
            if type(edges) is ctypes.c_void_p:
                edgepair_list[i] = edges
            elif type(edges) is np.ndarray:
                edgepair_list[i] = ctypes.c_void_p(edges.ctypes.data)
            else:
                raise NotImplementedError

            list_num_nodes[i] = graph_list[i].num_nodes
            list_num_edges[i] = edges.shape[0] // 2
        total_num_nodes = np.sum(list_num_nodes)
        total_num_edges = np.sum(list_num_edges)

        edge_to_idx = torch.LongTensor(total_num_edges)
        edge_from_idx = torch.LongTensor(total_num_edges)
        g_idx = torch.LongTensor(total_num_nodes)
        self.lib.prepare_indices(len(graph_list), 
                                 ctypes.c_void_p(list_num_nodes.ctypes.data),
                                 ctypes.c_void_p(list_num_edges.ctypes.data),
                                 ctypes.cast(edgepair_list, ctypes.c_void_p),
                                 ctypes.c_void_p(edge_to_idx.numpy().ctypes.data),
                                 ctypes.c_void_p(edge_from_idx.numpy().ctypes.data),
                                 ctypes.c_void_p(g_idx.numpy().ctypes.data))
        return edge_to_idx.to(device), edge_from_idx.to(device), g_idx.to(device)

    def PrepareMeanField(self, s2v_graphs, device):
        n2n_sp_list = []
        single_graph = self._getGraphOrList(s2v_graphs)
        if single_graph is not None:
            n_etypes = len(single_graph.typed_edge_list)
        else:
            n_etypes = len(s2v_graphs[0].typed_edge_list)
        for i in range(n_etypes):
            if single_graph is not None:
                n2n_sp = single_graph.n2n_sp_list[i]
            else:
                num_edges = 0
                for g in s2v_graphs:
                    num_edges += len(g.typed_edge_list[i])                
                n2n_idxes = torch.LongTensor(2, num_edges)
                n2n_vals = torch.FloatTensor(num_edges)

                num_nodes = 0
                nnz = 0
                for j in range(len(s2v_graphs)):
                    g = s2v_graphs[j]
                    n2n_idxes[:, nnz : nnz + len(g.typed_edge_list[i])] = g.n2n_sp_list[i]._indices() + num_nodes
                    n2n_vals[nnz : nnz + len(g.typed_edge_list[i])] = g.n2n_sp_list[i]._values()
                    num_nodes += g.pg.num_nodes
                    nnz += len(g.typed_edge_list[i])
                assert nnz == num_edges

                n2n_sp = torch.sparse.FloatTensor(n2n_idxes, n2n_vals, torch.Size([num_nodes, num_nodes])).to(device)
            
            n2n_sp_list.append(n2n_sp)

        return n2n_sp_list

    def ConcatFeats(self, s2v_graphs, device, feat_fn = lambda x: x.node_feat):
        single_graph = self._getGraphOrList(s2v_graphs)

        if single_graph is not None:
            feat = feat_fn(single_graph)
        else:
            feat_list = []
            for g in s2v_graphs:
                feat_list.append(feat_fn(g))
            
            feat = torch.cat(feat_list, dim=0).to(device)
        return Variable(feat)

dll_path = '%s/build/dll/libs2v.so' % os.path.dirname(os.path.realpath(__file__))
if os.path.exists(dll_path):
    S2VLIB = _s2v_lib(sys.argv)
else:
    S2VLIB = None


class Graph4NN(object):
    def __init__(self, pg):
        super(Graph4NN, self).__init__()
        self.label = pg.label
        self.pg = pg
        self.num_nodes = pg.num_nodes
        self.num_edges = len(pg.edge_list)
        self.typed_edge_list = pg.typed_edge_list
        self.target_idx = pg.target_idx
        self.node_feat = torch.zeros(pg.num_nodes, pg.num_node_feats, dtype=t_float)

        for i in range(pg.num_nodes):
            self.node_feat[i, pg.node_list[i]] = 1.0


class Code2InvGraph(Graph4NN):
    def __init__(self, pg):
        super(Code2InvGraph, self).__init__(pg)
                
        self.n2n_sp_list = []
        for i in range(len(self.typed_edge_list)):
            edges = self.typed_edge_list[i]
            degrees = np.zeros(shape=(pg.num_nodes), dtype=np.int32)
            for e in edges:
                degrees[e[1]] += 1                            
            edges.sort(key = lambda x : (x[1], x[0]))
            
            num_edges = len(edges)
            n2n_idxes = torch.LongTensor(2,  num_edges)
            n2n_vals = torch.FloatTensor(num_edges)

            edge_pairs = np.ndarray(shape=(num_edges, 2), dtype=np.int32)
            if num_edges:
                x, y = zip(*edges)            
                edge_pairs[:, 0] = x
                edge_pairs[:, 1] = y
            edge_pairs = edge_pairs.flatten()
            
            S2VLIB.lib.n2n_construct(pg.num_nodes,
                                     num_edges,
                                     ctypes.c_void_p(degrees.ctypes.data),
                                     ctypes.c_void_p(edge_pairs.ctypes.data),
                                     ctypes.c_void_p(n2n_idxes.numpy().ctypes.data),
                                     ctypes.c_void_p(n2n_vals.numpy().ctypes.data))

            n2n_sp = torch.sparse.FloatTensor(n2n_idxes, n2n_vals, torch.Size([pg.num_nodes, pg.num_nodes]))
            
            self.n2n_sp_list.append(n2n_sp)


class MultiGraph(Graph4NN):
    def __init__(self, pg):
        super(MultiGraph, self).__init__(pg)

        self.list_edge_pairs = []
        for i in range(len(self.typed_edge_list)):
            edges = self.typed_edge_list[i]

            edge_pairs = np.ndarray(shape=(len(edges), 2), dtype=np.int32)
            if len(edges):
                x, y = zip(*edges)
                edge_pairs[:, 0] = x
                edge_pairs[:, 1] = y
            edge_pairs = edge_pairs.flatten()
            self.list_edge_pairs.append(edge_pairs)


class MergedGraph(Graph4NN):
    def __init__(self, pg):
        super(MergedGraph, self).__init__(pg)

        self.edge_pairs = np.zeros((self.num_edges * 2, ), dtype=np.int32)
        self.edge_feat = torch.zeros(self.num_edges, len(self.typed_edge_list), dtype=t_float)

        for i, e in enumerate(self.pg.edge_list):
            self.edge_pairs[i * 2] = e[0]
            self.edge_pairs[i * 2 + 1] = e[1]
            self.edge_feat[i, e[2]] = 1.0


def get_gnn_graph(code_graph, gnn_type):
    if gnn_type == 's2v_code2inv':
        return Code2InvGraph(code_graph)
    elif gnn_type == 's2v_multi':
        return MultiGraph(code_graph)
    else:
        return MergedGraph(code_graph)