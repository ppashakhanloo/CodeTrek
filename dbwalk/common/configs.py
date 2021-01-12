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


# data process
cmd_opt.add_argument('-data_chunk_size', default=1, type=int, help='# samples per data file')

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
