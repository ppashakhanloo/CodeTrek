UNK = '__unknown__'
EOS = '__eos__'
TOK_PAD = '__pad__'

DUMMY_ARRAY = [[[0, 0], [0, 0]]]

import torch

t_float = torch.float32

def var_idx2name(idx):
    return 'var_%d' % idx
