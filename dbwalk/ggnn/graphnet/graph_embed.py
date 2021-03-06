from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import sys
import numpy as np
import torch
import json
import random
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dbwalk.common.pytorch_util import gnn_spmm, MLP, _param_init
from dbwalk.ggnn.graphnet.s2v_lib import S2VLIB

from torch_geometric.nn.conv import MessagePassing

from dbwalk.ggnn.graphnet.utils import get_agg, ReadoutNet, NONLINEARITIES, scatter_add


class GraphNN(nn.Module):
    def __init__(self, latent_dim, output_dim, num_node_feats, num_edge_feats, num_node_val_feats, max_lv, act_func, readout_agg, gnn_out):
        super(GraphNN, self).__init__()
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.num_node_val_feats = num_node_val_feats
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.max_lv = max_lv
        self.act_func = NONLINEARITIES[act_func]

        self.node_val_embedding = Parameter(torch.Tensor(num_node_val_feats, latent_dim))
        _param_init(self.node_val_embedding)
        self.readout_net = ReadoutNet(node_state_dim=latent_dim,
                                      output_dim=output_dim,
                                      max_lv=max_lv,
                                      act_func=act_func,
                                      out_method=gnn_out,
                                      readout_agg=readout_agg,
                                      act_last=True,
                                      bn=False)
    @property
    def device(self):
        return next(self.parameters()).device


class _MeanFieldLayer(MessagePassing):
    def __init__(self, latent_dim, param_update=True):
        super(_MeanFieldLayer, self).__init__()
        self.param_update = param_update
        if param_update:
            self.conv_params = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, edge_index):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, x=x)

    def update(self, aggr_out):
        return self.conv_params(aggr_out) if self.param_update else aggr_out


class S2VSingle(GraphNN):
    def __init__(self, latent_dim, output_dim, num_node_feats, num_edge_feats, num_node_val_feats, max_lv=3, act_func='relu', readout_agg='max', gnn_out='last'):
        super(S2VSingle, self).__init__(latent_dim, output_dim, num_node_feats, num_edge_feats, num_node_val_feats, max_lv, act_func, readout_agg, gnn_out)

        self.w_n2l = nn.Linear(num_node_feats, latent_dim)
        if self.num_edge_feats > 0:
            self.w_e2l = [nn.Linear(self.num_edge_feats, latent_dim) for _ in range(self.max_lv + 1)]
            self.w_e2l = nn.ModuleList(self.w_e2l)

        lm_layer = lambda: _MeanFieldLayer(latent_dim)        
        conv_layers = [lm_layer() for _ in range(max_lv)]
        self.conv_layers = nn.ModuleList(conv_layers)
        self.conv_l2 = [nn.Linear(latent_dim, latent_dim) for _ in range(self.max_lv)]
        self.conv_l2 = nn.ModuleList(self.conv_l2)

        msg_bn = [nn.BatchNorm1d(latent_dim) for _ in range(self.max_lv + 1)]
        hidden_bn = [nn.BatchNorm1d(latent_dim) for _ in range(self.max_lv)]
        self.msg_bn = nn.ModuleList(msg_bn)
        self.hidden_bn = nn.ModuleList(hidden_bn)

    def forward(self, graph_list, node_val_mat):
        device = self.device
        node_feat = S2VLIB.ConcatFeats(graph_list, device)
        input_node_linear = self.w_n2l(node_feat)
        node_embed_feat = gnn_spmm(node_val_mat, self.node_val_embedding)
        input_node_linear = input_node_linear + node_embed_feat
        edge_from_idx, edge_to_idx, g_idx = S2VLIB.PrepareIndices(graph_list, device)
        input_message = input_node_linear
        if self.num_edge_feats > 0:
            edge_feat = S2VLIB.ConcatFeats(graph_list, device, feat_fn=lambda x: x.edge_feat)
            input_edge_linear = self.w_e2l[0](edge_feat)
            e2npool_input = scatter_add(input_edge_linear, edge_to_idx, dim=0, dim_size=node_feat.shape[0])
            input_message += e2npool_input
        input_potential = self.act_func(input_message)
        input_potential = self.msg_bn[0](input_potential)

        cur_message_layer = input_potential
        all_embeds = [cur_message_layer]
        edge_index = torch.stack([edge_from_idx, edge_to_idx], dim=0)
        for lv in range(self.max_lv):
            node_linear = self.conv_layers[lv](cur_message_layer, edge_index)
            edge_linear = self.w_e2l[lv + 1](edge_feat)
            e2npool_input = scatter_add(edge_linear, edge_to_idx, dim=0, dim_size=node_linear.shape[0])
            merged_hidden = self.act_func(node_linear + e2npool_input)
            merged_hidden = self.hidden_bn[lv](merged_hidden)
            residual_out = self.conv_l2[lv](merged_hidden) + cur_message_layer
            cur_message_layer = self.act_func(residual_out)
            cur_message_layer = self.msg_bn[lv + 1](cur_message_layer)
            all_embeds.append(cur_message_layer)
        return self.readout_net(all_embeds, g_idx, len(graph_list))


class S2VMulti(GraphNN):
    def __init__(self, latent_dim, output_dim, num_node_feats, num_edge_feats, num_node_val_feats,
                max_lv=3, readout_agg='max', act_func='tanh', gnn_out='last'):
        super(S2VMulti, self).__init__(latent_dim, output_dim, num_node_feats, num_edge_feats, num_node_val_feats, max_lv, act_func, readout_agg, gnn_out)
        self.readout_agg = get_agg(readout_agg)

        self.w_n2l = nn.Linear(num_node_feats, latent_dim)
        self.conv_param_list = []
        self.merge_param_list = []
        for i in range(self.max_lv):
            self.conv_param_list.append(nn.Linear(latent_dim, self.num_edge_feats * latent_dim))
            self.merge_param_list.append( nn.Linear(self.num_edge_feats * latent_dim, latent_dim) )
        self.conv_param_list = nn.ModuleList(self.conv_param_list)
        self.merge_param_list = nn.ModuleList(self.merge_param_list)

        lm_layer = lambda: _MeanFieldLayer(latent_dim, param_update=False)
        conv_layers = [lm_layer() for _ in range(max_lv)]
        self.conv_layers = nn.ModuleList(conv_layers)
        self.conv_l2 = [nn.Linear(latent_dim, latent_dim) for _ in range(self.max_lv)]
        self.conv_l2 = nn.ModuleList(self.conv_l2)

    def _forward(self, input_node_linear, graph_list):
        device = self.device
        list_edge_idx = []
        for i in range(self.num_edge_feats):
            edge_from_idx, edge_to_idx, g_idx = S2VLIB.PrepareIndices(graph_list, device, fn_edges=lambda x: x.list_edge_pairs[i])            
            list_edge_idx.append(torch.stack((edge_from_idx, edge_to_idx), dim=0))

        input_message = input_node_linear
        input_potential = self.act_func(input_message)

        cur_message_layer = input_potential
        all_embeds = [cur_message_layer]

        for lv in range(self.max_lv):
            conv_feat = self.conv_param_list[lv](cur_message_layer)
            chunks = torch.split(conv_feat, self.latent_dim, dim=1)

            msg_list = []
            for i in range(self.num_edge_feats):
                t = self.conv_layers[lv](chunks[i], list_edge_idx[i])
                msg_list.append(t)
            msg = self.act_func( torch.cat(msg_list, dim=1) )
            merged_hidden = self.merge_param_list[lv](msg)
            residual_out = self.conv_l2[lv](merged_hidden) + cur_message_layer
            cur_message_layer = self.act_func(residual_out)
            all_embeds.append(cur_message_layer)
        return self.readout_net(all_embeds, g_idx, len(graph_list))

    def forward(self, graph_list, node_val_mat):
        device = self.device
        node_feat = S2VLIB.ConcatFeats(graph_list, device)
        node_embed_feat = gnn_spmm(node_val_mat, self.node_val_embedding)
        input_node_linear = self.w_n2l(node_feat) + node_embed_feat
        return self._forward(input_node_linear, graph_list)


class Code2InvMulti(GraphNN):
    def __init__(self, latent_dim, output_dim, num_node_feats, num_edge_feats, num_node_val_feats, max_lv=3, readout_agg='max', act_func='tanh', gnn_out='last'):
        super(Code2InvMulti, self).__init__(latent_dim, output_dim, num_node_feats, num_edge_feats, num_node_val_feats, max_lv, act_func, readout_agg, gnn_out)
        self.readout_agg = get_agg(readout_agg)

        self.w_n2l = nn.Linear(num_node_feats, latent_dim)
        self.conv_param_list = []
        self.merge_param_list = []
        for i in range(self.max_lv):
            self.conv_param_list.append(nn.Linear(latent_dim, self.num_edge_feats * latent_dim))
            self.merge_param_list.append( nn.Linear(self.num_edge_feats * latent_dim, latent_dim) )

        self.conv_param_list = nn.ModuleList(self.conv_param_list)
        self.merge_param_list = nn.ModuleList(self.merge_param_list)

    def forward(self, graph_list, node_val_mat):
        device = self.device
        node_feat = S2VLIB.ConcatFeats(graph_list, device)
        sp_list = S2VLIB.PrepareMeanField(graph_list, device)
        input_node_linear = self.w_n2l(node_feat)
        node_embed_feat = gnn_spmm(node_val_mat, self.node_val_embedding)
        input_node_linear = input_node_linear + node_embed_feat
        h = self.mean_field(input_node_linear, sp_list)
        g_idx = []
        for i, g in enumerate(graph_list):
            g_idx += [i] * g.pg.num_nodes
        g_idx = torch.LongTensor(g_idx).to(h[0].device)

        return self.readout_net(h, g_idx, len(graph_list))

    def mean_field(self, input_node_linear, sp_list):        
        input_message = input_node_linear
        input_potential = self.act_func(input_message)

        lv = 0
        cur_message_layer = input_potential
        all_embeds = [cur_message_layer]
        while lv < self.max_lv:
            conv_feat = self.conv_param_list[lv](cur_message_layer)
            chunks = torch.split(conv_feat, self.latent_dim, dim=1)
            
            msg_list = []
            for i in range(self.num_edge_feats):
                t = gnn_spmm(sp_list[i], chunks[i])
                msg_list.append( t )
            
            msg = self.act_func( torch.cat(msg_list, dim=1) )
            cur_input = self.merge_param_list[lv](msg)

            cur_message_layer = cur_input + cur_message_layer
            cur_message_layer = self.act_func(cur_message_layer)
            all_embeds.append(cur_message_layer)
            lv += 1

        return all_embeds


class GGNN(Code2InvMulti):
    def __init__(self, latent_dim, output_dim, num_node_feats, num_edge_feats, num_name_feats, max_lv=3, readout_agg='max', act_func='tanh'):
        assert num_name_feats is not None
        super(GGNN, self).__init__(latent_dim, output_dim, num_node_feats, num_edge_feats, max_lv=max_lv, readout_agg=readout_agg, act_func=act_func, gnn_out='gru')
        self.w_name2l = nn.Linear(num_name_feats, latent_dim)
        self.merge_feat = MLP(2 * latent_dim, [2 * latent_dim, latent_dim], nonlinearity=act_func)

    def forward(self, graph_list):
        device = self.device
        node_feat = S2VLIB.ConcatFeats(graph_list, device)
        name_feat = S2VLIB.ConcatFeats(graph_list, device, feat_fn=lambda x: x.name_feat)
        node_feat = self.w_n2l(node_feat)
        name_feat = self.w_name2l(name_feat)
        feat = self.act_func(torch.cat([node_feat, name_feat], dim=-1))
        input_node_linear = self.merge_feat(feat)

        sp_list = S2VLIB.PrepareMeanField(graph_list, device)
        h = self.mean_field(input_node_linear, sp_list)
        g_idx = []
        for i, g in enumerate(graph_list):
            g_idx += [i] * g.pg.num_nodes
        g_idx = torch.LongTensor(g_idx).to(h[0].device)

        return self.readout_net(h, g_idx, len(graph_list))


def get_gnn(args, node_dim, edge_dim, node_val_dim):
    if args.gnn_type == 's2v_code2inv':
        gnn = Code2InvMulti(args.gnn_msg_dim,
                            args.latent_dim,
                            node_dim,
                            edge_dim,
                            node_val_dim,
                            max_lv=args.max_lv,
                            act_func=args.act_func)
    elif args.gnn_type == 's2v_single':
        gnn = S2VSingle(args.gnn_msg_dim,
                        args.latent_dim,
                        node_dim,
                        edge_dim,
                        node_val_dim,
                        max_lv=args.max_lv,
                        act_func=args.act_func,
                        readout_agg=args.readout_agg_type,
                        gnn_out=args.gnn_out)
    elif args.gnn_type == 's2v_multi':
        gnn = S2VMulti(latent_dim=args.gnn_msg_dim,
                       output_dim=args.latent_dim,
                       num_node_feats=node_dim,
                       num_edge_feats=edge_dim,
                       num_node_val_feats=node_val_dim,
                       max_lv=args.max_lv,
                       act_func=args.act_func,
                       readout_agg=args.readout_agg_type,
                       gnn_out=args.gnn_out)
    elif args.gnn_type == 'ggnn':
        gnn = GGNN(latent_dim=args.gnn_msg_dim,
                   output_dim=args.latent_dim,
                   num_node_feats=node_dim,
                   num_edge_feats=edge_dim,
                   num_name_feats=name_dim,
                   max_lv=args.max_lv,
                   act_func=args.act_func,
                   readout_agg=args.readout_agg_type)
    else:
        raise NotImplementedError
    return gnn
