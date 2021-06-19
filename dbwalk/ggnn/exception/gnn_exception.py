from __future__ import print_function, division

import os
import sys
import torch
import numpy as np
import random
from functools import partial

from dbwalk.data_util.dataset import ProgDict
from dbwalk.common.configs import cmd_args, get_torch_device
from dbwalk.training.train import train_loop, multiclass_eval_dataset
from dbwalk.ggnn.graphnet.classifier import GnnMulticlass, gnn_eval_nn_args, gnn_arg_constructor
from dbwalk.ggnn.data_util.graph_dataset import AstGraphDataset


if __name__ == '__main__':
    np.random.seed(cmd_args.seed)
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    prog_dict = ProgDict(cmd_args.data_dir)

    model = GnnMulticlass(cmd_args, prog_dict, has_anchor=False)
    eval_func = partial(multiclass_eval_dataset, fn_parse_eval_nn_args=gnn_eval_nn_args)
    if cmd_args.phase == 'eval': 
        db_eval = AstGraphDataset(cmd_args, prog_dict, cmd_args.data_dir, 'eval')
        eval_loader = db_eval.get_test_loader(cmd_args)
        assert cmd_args.model_dump is not None
        model_dump = os.path.join(cmd_args.save_dir, cmd_args.model_dump)
        print('loading model from', model_dump)
        model.load_state_dict(torch.load(model_dump, map_location=get_torch_device(cmd_args.gpu)))
        eval_func(model, 'eval', eval_loader, get_torch_device(cmd_args.gpu))
        sys.exit()

    db_dev = AstGraphDataset(cmd_args, prog_dict, cmd_args.data_dir, 'dev')
    db_train = AstGraphDataset(cmd_args, prog_dict, cmd_args.data_dir, 'train')
    train_loop(cmd_args, get_torch_device(cmd_args.gpu), prog_dict, model, db_train, db_dev, eval_func,
               nn_arg_constructor=gnn_arg_constructor)
