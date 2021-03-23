import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import ProgWalkTokEmbed, ProgWalkTokEmbedWithVal, ProgWalkEncoder, ProgDeepset, ProgTransformer

from dbwalk.common.pytorch_util import MLP


class WalkSet2Embed(nn.Module):
    def __init__(self, args, prog_dict):
        super(WalkSet2Embed, self).__init__()
        self.use_node_val = args.use_node_val
        if self.use_node_val:
            self.tok_encoding = ProgWalkTokEmbedWithVal(prog_dict, args.embed_dim, args.dropout)
        else:
            self.tok_encoding = ProgWalkTokEmbed(prog_dict, args.embed_dim, args.dropout)
        self.walk_encoding = ProgWalkEncoder(args.embed_dim, args.nhead, args.transformer_layers,  args.dim_feedforward, args.dropout)
        if args.set_encoder == 'deepset':
            self.prob_encoding = ProgDeepset(args.embed_dim, args.dropout)
        elif args.set_encoder == 'transformer':
            self.prob_encoding = ProgTransformer(args.embed_dim, args.nhead, args.transformer_layers,  args.dim_feedforward, args.dropout)
        else:
            raise ValueError("unknown set encoder %s" % args.set_encoder)

    def forward(self, node_idx, edge_idx, node_val_mat=None):
        if self.use_node_val:
            assert node_val_mat is not None
            seq_tok_embed = self.tok_encoding(node_idx, edge_idx, node_val_mat)
        else:
            seq_tok_embed = self.tok_encoding(node_idx, edge_idx)
        walk_repr = self.walk_encoding(seq_tok_embed)
        prog_repr = self.prob_encoding(walk_repr)
        return prog_repr


class BinaryNet(WalkSet2Embed):
    def __init__(self, args, prog_dict):
        super(BinaryNet, self).__init__(args, prog_dict)
        assert prog_dict.num_class == 2
        self.out_classifier = nn.Linear(args.embed_dim, 1)

    def forward(self, node_idx, edge_idx, *, node_val_mat=None, label=None):
        prog_repr = super(BinaryNet, self).forward(node_idx, edge_idx, node_val_mat)

        logits = self.out_classifier(prog_repr)
        prob = torch.sigmoid(logits)
        if label is not None:
            label = label.to(prob).view(prob.shape)
            loss = -label * torch.log(prob + 1e-18) - (1 - label) * torch.log(1 - prob + 1e-18)
            return torch.mean(loss)
        else:
            return prob


class MulticlassNet(WalkSet2Embed):
    def __init__(self, args, prog_dict):
        super(MulticlassNet, self).__init__(args, prog_dict)
        #self.out_classifier = MLP(args.embed_dim, [args.embed_dim * 2, prog_dict.num_class])
        self.out_classifier = nn.Linear(args.embed_dim, prog_dict.num_class)

    def forward(self, node_idx, edge_idx, *, node_val_mat=None, label=None):
        prog_repr = super(MulticlassNet, self).forward(node_idx, edge_idx, node_val_mat)
        logits = self.out_classifier(prog_repr)

        if label is not None:
            label = label.to(logits.device).view(logits.shape[0])
            loss = F.cross_entropy(logits, label)
            return loss
        else:
            return logits
