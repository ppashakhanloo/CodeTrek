UNK = '__unknown__'
EOS = '__eos__'
TOK_PAD = '__pad__'

import torch

t_float = torch.float32

def var_idx2name(idx):
    return 'var_%d' % idx