from os import walk
from dbwalk.data_util.cook_data import get_or_unk
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from dbwalk.common.pytorch_util import PositionalEncoding, MLP
from dbwalk.common.pytorch_util import gnn_spmm, _param_init


class PreNormTransformer(TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(PreNormTransformer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = self.norm1(src)
        x = self.self_attn(x, x, x, attn_mask=src_mask,
                           key_padding_mask=src_key_padding_mask)[0]
        x = self.dropout1(x) + src
        z = self.norm2(x)
        z = self.linear2(self.dropout(self.activation(self.linear1(z))))
        z = x + self.dropout2(z)
        return z


class ProgWalkTokEmbed(nn.Module):
    def __init__(self, prog_dict, embed_dim, dropout=0.0):
        super(ProgWalkTokEmbed, self).__init__()
        self.pos_encoding = PositionalEncoding(d_model=embed_dim, dropout=dropout)
        self.pg_dict = prog_dict

        self.node_embed = nn.Embedding(self.pg_dict.num_node_types, embed_dim)
        self.edge_embed = nn.Embedding(self.pg_dict.num_edge_types, embed_dim)

    def forward(self, node_idx, edge_idx):
        node_embed = self.node_embed(node_idx)
        edge_embed = self.edge_embed(edge_idx)
        node_embed = self.pos_encoding(node_embed)
        edge_embed = self.pos_encoding(edge_embed)
        return torch.cat((node_embed, edge_embed), dim=0)


class ProgWalkTokEmbedWithVal(ProgWalkTokEmbed):
    def __init__(self, prog_dict, embed_dim, dropout=0.0):
        super(ProgWalkTokEmbedWithVal, self).__init__(prog_dict, embed_dim, dropout)
        self.val_tok_embed = Parameter(torch.Tensor(self.pg_dict.num_node_val_tokens, embed_dim))
        _param_init(self.val_tok_embed)

    def forward(self, node_idx, edge_idx, node_val_mat):
        node_embed = self.node_embed(node_idx)
        edge_embed = self.edge_embed(edge_idx)
 
        node_val_embed = gnn_spmm(node_val_mat, self.val_tok_embed).view(node_embed.shape)
        node_embed = self.pos_encoding(node_embed)
        edge_embed = self.pos_encoding(edge_embed)
        val_embed = self.pos_encoding(node_val_embed)
        return torch.cat((node_embed, edge_embed, val_embed), dim=0)


class ProgWalkEncoder(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 4, num_encoder_layers: int = 3, 
                 dim_feedforward: int = 512, dropout: float = 0.0, activation: str = "relu"):
        super(ProgWalkEncoder, self).__init__()
        self.d_model = d_model
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    def forward(self, walk_token_embed):
        assert walk_token_embed.dim() == 4 # L x N x B x d_model
        L, N, B, _ = walk_token_embed.shape
        walk_token_embed = walk_token_embed.view(L, -1, self.d_model)

        memory = self.encoder(walk_token_embed)
        memory = memory.view(L, N, B, -1)
        walk_repr = torch.mean(memory, dim=0)
        return walk_repr


class ProgDeepset(nn.Module):
    def __init__(self, embed_dim, dropout=0.0):
        super(ProgDeepset, self).__init__()
        self.mlp = MLP(embed_dim, [2 * embed_dim, embed_dim], dropout=dropout)

    def forward(self, walk_repr, get_before_agg=False):
        walk_hidden = self.mlp(walk_repr)
        prog_repr, _ = torch.max(walk_hidden, dim=0)
        #prog_repr = torch.mean(walk_hidden, dim=0)
        if get_before_agg:
            return prog_repr, walk_hidden
        else:
            return prog_repr


class ProgTransformer(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 4, num_encoder_layers: int = 3, 
                 dim_feedforward: int = 512, dropout: float = 0.0, activation: str = "relu"):
        super(ProgTransformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    def forward(self, walk_embed, get_before_agg=False):
        assert walk_embed.dim() == 3 # N x B x d_model
        assert not get_before_agg
        memory = self.encoder(walk_embed)
        return memory[0]


if __name__ == '__main__':
    pass
