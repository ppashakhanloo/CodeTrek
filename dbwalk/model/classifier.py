import torch
import torch.nn as nn

from .encoder import ProgWalkTokEmbed, ProgWalkEncoder, ProgDeepset


class BinaryNet(nn.Module):
    def __init__(self, args, prog_dict):
        super(BinaryNet, self).__init__()
        self.tok_encoding = ProgWalkTokEmbed(prog_dict, args.embed_dim, args.dropout)
        self.walk_encoding = ProgWalkEncoder(args.embed_dim, args.nhead, args.transformer_layers,  args.dim_feedforward, args.dropout)
        self.prob_encoding = ProgDeepset(args.embed_dim, args.dropout)

        self.out_classifier = nn.Linear(args.embed_dim, 1)

    def forward(self, node_idx, edge_idx, label=None):
        seq_tok_embed = self.tok_encoding(node_idx, edge_idx)        
        walk_repr = self.walk_encoding(seq_tok_embed)
        prog_repr = self.prob_encoding(walk_repr)
        
        logits = self.out_classifier(prog_repr)
        prob = torch.sigmoid(logits)
        if label is not None:
            label = label.to(prob).view(prob.shape)
            loss = -label * torch.log(prob + 1e-18) - (1 - label) * torch.log(1 - prob + 1e-18)
            return torch.mean(loss)
        else:
            return prob