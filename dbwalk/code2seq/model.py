import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from dbwalk.common.pytorch_util import gnn_spmm, _param_init


class Code2seqTokEmbedWithVal(nn.Module):
    def __init__(self, prog_dict, embed_dim, dropout=0.0):
        super(Code2seqTokEmbedWithVal, self).__init__()
        self.node_embed = nn.Embedding(prog_dict.num_node_types, embed_dim)
        self.val_tok_embed = Parameter(torch.Tensor(prog_dict.num_node_val_tokens, embed_dim))
        _param_init(self.val_tok_embed)

    def forward(self, node_idx, node_val_mat):
        node_embed = self.node_embed(node_idx)
        _, N, B, e = node_embed.shape
        node_val_embed = gnn_spmm(node_val_mat, self.val_tok_embed).view(-1, N, B, e)
        return node_embed, node_val_embed


class Code2seqEncoder(nn.Module):
    def __init__(self, n_layers, embed_dim, dropout=0.0):
        super(Code2seqEncoder, self).__init__()
        self.embed_dim = embed_dim
        
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=embed_dim,
                            num_layers=n_layers,
                            bidirectional=True)
        self.proj = nn.Linear(embed_dim * 4, embed_dim)

    def forward(self, node_embed, node_val_embed):        
        L, N, B, _ = node_embed.shape
        node_embed = node_embed.view(L, -1, self.embed_dim)

        _, state = self.lstm(node_embed)
        state = state[0].view(self.n_layers, 2, N*B, self.embed_dim)[-1].view(2, N, B, self.embed_dim)
        
        joint_state = torch.split(state, 1, dim=0) + torch.split(node_val_embed, 1, dim=0)
        joint_state = torch.cat([x.squeeze(0) for x in joint_state], dim=-1)
        
        path_state = torch.tanh(self.proj(joint_state))
        return torch.mean(path_state, dim=0)


class BinaryCode2seqNet(nn.Module):
    def __init__(self, args, prog_dict):
        super(BinaryCode2seqNet, self).__init__()
        assert prog_dict.num_class == 2
        self.tok_encoder = Code2seqTokEmbedWithVal(prog_dict, args.embed_dim)

        self.out_classifier = nn.Linear(args.embed_dim, 1)
        self.ctx_encoder = Code2seqEncoder(args.transformer_layers, args.embed_dim)

    def forward(self, node_idx, edge_idx, *, node_val_mat, label=None):
        assert edge_idx is None
        node_embed, node_val_embed = self.tok_encoder(node_idx, node_val_mat)
        prog_repr = self.ctx_encoder(node_embed, node_val_embed)

        logits = self.out_classifier(prog_repr)
        prob = torch.sigmoid(logits)
        if label is not None:
            label = label.to(prob).view(prob.shape)
            loss = -label * torch.log(prob + 1e-18) - (1 - label) * torch.log(1 - prob + 1e-18)
            return torch.mean(loss)
        else:
            return prob
