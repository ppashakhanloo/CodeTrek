from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
import os
import pickle as cp
import torch

cmd_opt = argparse.ArgumentParser(description='Argparser for dbwalk', allow_abbrev=False)
cmd_opt.add_argument('-save_dir', default='.', help='result output root')
cmd_opt.add_argument('-data_dir', default='.', help='data dir')
cmd_opt.add_argument('-data', default=None, help='data name')

cmd_opt.add_argument('-phase', default='train', help='train/test')
cmd_opt.add_argument('-model_dump', default=None, help='load model dump')

cmd_opt.add_argument('-gpu', type=int, default=-1, help='-1: cpu; 0 - ?: specific gpu index')
cmd_opt.add_argument('-num_proc', type=int, default=1, help='number of processes')

# transformer
cmd_opt.add_argument('-embed_dim', default=256, type=int, help='embed size')
cmd_opt.add_argument('-nhead', default=4, type=int, help='multi-head attention')
cmd_opt.add_argument('-transformer_layers', default=3, type=int, help='# transformer layers')
cmd_opt.add_argument('-dropout', default=0, type=float, help='dropout')
cmd_opt.add_argument('-dim_feedforward', default=512, type=int, help='embed size')

cmd_opt.add_argument('-seed', default=19260817, type=int, help='seed')
cmd_opt.add_argument('-learning_rate', default=1e-3, type=float, help='learning rate')
cmd_opt.add_argument('-grad_clip', default=5, type=float, help='gradient clip')

cmd_opt.add_argument('-num_epochs', default=100000, type=int, help='num epochs')
cmd_opt.add_argument('-batch_size', default=64, type=int, help='batch size')

cmd_opt.add_argument('-epoch_save', default=100, type=int, help='num epochs between save')
cmd_opt.add_argument('-iter_per_epoch', default=100, type=int, help='num iterations per epoch')

cmd_opt.add_argument('-epoch_load', default=None, type=int, help='epoch for loading')
cmd_opt.add_argument('-shuffle_var', default=False, type=eval, help='shuffle var index?')
cmd_opt.add_argument('-online_walk_gen', default=False, type=eval, help='generate random walks on the fly?')

cmd_opt.add_argument('-use_node_val', default=False, type=eval, help='use node value as feature?')
cmd_opt.add_argument('-set_encoder', default='deepset', help='deepset/transformer')

# data process
cmd_opt.add_argument('-data_chunk_size', default=1, type=int, help='# samples per data file')

# walks
cmd_opt.add_argument('-min_steps', default=1, type=int, help='min steps')
cmd_opt.add_argument('-max_steps', default=16, type=int, help='max steps')
cmd_opt.add_argument('-num_walks', default=100, type=int, help='number of random walks per file')
cmd_opt.add_argument('-language', default='python', type=str, help='language')


# gnn
cmd_opt.add_argument('-gnn_type', default='s2v_multi', help='type of graph neural network', choices=['s2v_code2inv', 's2v_single', 's2v_multi', 'ggnn'])
cmd_opt.add_argument('-rnn_cell', default='gru', help='type of rnn cell')
cmd_opt.add_argument('-act_func', default='tanh', help='default activation function')
cmd_opt.add_argument('-max_lv', default=3, type=int, help='# layers of gnn')
cmd_opt.add_argument('-msg_agg_type', default='sum', help='how to aggregate the message')
cmd_opt.add_argument('-att_type', default='inner_prod', help='mlp/inner_prod')
cmd_opt.add_argument('-readout_agg_type', default='sum', help='how to aggregate all node embeddings', choices=['sum', 'max', 'mean'])
cmd_opt.add_argument('-gnn_out', default='last', help='how to aggregate readouts from different layers', choices=['last', 'sum', 'max', 'gru', 'mean'])
cmd_opt.add_argument('-gnn_msg_dim', default=128, type=int, help='dim of message passing in gnn')
cmd_opt.add_argument('-latent_dim', default=128, type=int, help='latent dim')


cmd_args, _ = cmd_opt.parse_known_args()

if cmd_args.save_dir is not None:
    if not os.path.isdir(cmd_args.save_dir):
        os.makedirs(cmd_args.save_dir)

print(cmd_args)

def set_device(gpu):
    if torch.cuda.is_available() and gpu >= 0:
        cmd_args.gpu = gpu
        cmd_args.device = torch.device('cuda:' + str(gpu))
        print('use gpu indexed: %d' % gpu)
    else:
        cmd_args.gpu = -1
        cmd_args.device = torch.device('cpu')
        print('use cpu')
