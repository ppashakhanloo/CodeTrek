import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

# The contents of this file is mainly copied from 
# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py
# Then, it has been changed in two places:
#   1. TransformerEncoder forward returns output of the hidden layers and the output of
#      self-attention blocks in addition to its outputs in PyTorch.
#   2. In TransformerEncoderLayer, the output of the _sa_block is now a tuple of
#      (sa_output, sa_output_weights) instead of just the sa_output.
  

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class CustomTransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(CustomTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src

        hidden_vectors = []
        attn_weights_vectors = []
        for mod in self.layers:
            output, weight = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            hidden_vectors.append(output)
            attn_weights_vectors.append(weight)
        hidden_vectors = torch.stack(hidden_vectors)
        attn_weights_vectors = torch.stack(attn_weights_vectors)

        if self.norm is not None:
            output = self.norm(output)

        return output, hidden_vectors, attn_weights_vectors


class CustomTransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(CustomTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        if self.norm_first:
            sa = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + sa[0] #self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)[0]
            x = x + self._ff_block(self.norm2(x))
        else:
            sa = self._sa_block(x, src_mask, src_key_padding_mask)
            x = self.norm1(x + sa[0])  #self._sa_block(x, src_mask, src_key_padding_mask)[0])
            x = self.norm2(x + self._ff_block(x))

        return x, sa[1]

    # self-attention block
    def _sa_block(self, x,
                  attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True)
        # x0: attention_out, attention_out_weight
        return self.dropout1(x[0]), x[1]

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
