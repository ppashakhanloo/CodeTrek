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
from dbwalk.ggnn.graphnet.graph_embed import get_gnn


class GnnClassifierBase(nn.Module):
    def __init__(self, args, prog_dict, has_anchor):
        super(GnnClassifierBase, self).__init__()
        self.gnn = get_gnn(args, len(prog_dict.node_types), len(prog_dict.edge_types))
        self.has_anchor = has_anchor

    def get_embedding(self, graph_list):
        graph_embed, (_, node_embed) = self.gnn(graph_list)
        if self.has_anchor:
            node_sel = []
            offset = 0
            for g in graph_list:
                node_sel.append(g.target_idx)
                offset += g.num_nodes
            target_embed = node_embed[node_sel]
            return torch.cat((graph_embed, target_embed), dim=1)
        else:
            return graph_embed


class GnnBinary(GnnClassifierBase):
    def __init__(self, args, prog_dict, has_anchor):
        super(GnnBinary, self).__init__(args, prog_dict, has_anchor)
        self.out_classifier = nn.Linear(args.latent_dim * (2 if has_anchor else 1), 1)

    def forward(self, graph_list, label=None):
        state_repr = self.get_embedding(graph_list)
        logits = self.out_classifier(state_repr)
        prob = torch.sigmoid(logits)
        if label is not None:
            label = label.to(prob).view(prob.shape)
            loss = -label * torch.log(prob + 1e-18) - (1 - label) * torch.log(1 - prob + 1e-18)
            return torch.mean(loss)
        else:
            return prob


class GnnMulticlass(GnnClassifierBase):
    def __init__(self, args, prog_dict, has_anchor):
        super(GnnMulticlass, self).__init__(args, prog_dict, has_anchor)
        self.out_classifier = nn.Linear(args.latent_dim * (2 if has_anchor else 1), prog_dict.num_class)

    def forward(self, graph_list, label=None):
        state_repr = self.get_embedding(graph_list)
        logits = self.out_classifier(state_repr)
        if label is not None:
            label = label.to(logits.device).view(logits.shape[0])
            loss = F.cross_entropy(logits, label)
            return loss
        else:
            return logits
