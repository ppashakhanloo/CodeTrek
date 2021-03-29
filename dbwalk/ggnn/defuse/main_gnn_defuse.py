from __future__ import print_function, division

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm import tqdm
import numpy as np
import random
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from dbwalk.data_util.dataset import ProgDict
from dbwalk.ggnn.data_util.graph_dataset import AstGraphDataset
from dbwalk.common.configs import cmd_args, set_device

from dbwalk.ggnn.graphnet.classifier import GnnBinary, gnn_arg_constructor, gnn_eval_nn_args
from dbwalk.training.train import train_loop, binary_eval_dataset


if __name__ == '__main__':
    set_device(cmd_args.gpu)
    np.random.seed(cmd_args.seed)
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    prog_dict = ProgDict(cmd_args.data_dir)

    model = GnnBinary(cmd_args, prog_dict, has_anchor=False).to(cmd_args.device)
    eval_func = partial(binary_eval_dataset, fn_parse_eval_nn_args=gnn_eval_nn_args)
    if cmd_args.phase == 'eval': 
        db_eval = AstGraphDataset(cmd_args, prog_dict, cmd_args.data_dir, 'eval')
        eval_loader = db_eval.get_test_loader(cmd_args)
        assert cmd_args.model_dump is not None
        model_dump = os.path.join(cmd_args.save_dir, cmd_args.model_dump)
        print('loading model from', model_dump)
        model.load_state_dict(torch.load(model_dump, map_location=cmd_args.device))    
        eval_func(model, 'eval', eval_loader)
        sys.exit()

    db_dev = AstGraphDataset(cmd_args, prog_dict, cmd_args.data_dir, 'dev')
    db_train = AstGraphDataset(cmd_args, prog_dict, cmd_args.data_dir, 'train', sample_prob={'used': 0.5, 'unused': 0.5})
    train_loop(prog_dict, model, db_train, db_dev, eval_func,
               nn_arg_constructor=gnn_arg_constructor)
