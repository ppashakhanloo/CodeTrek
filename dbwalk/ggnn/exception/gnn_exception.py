from __future__ import print_function, division

import os
import sys
import torch
import numpy as np
import random
from functools import partial

from dbwalk.data_util.dataset import ProgDict
from dbwalk.common.configs import args, set_device
from dbwalk.training.train import train_loop, multiclass_eval_dataset
from dbwalk.ggnn.graphnet.classifier import GnnMulticlass, gnn_eval_nn_args, gnn_arg_constructor
from dbwalk.ggnn.data_util.graph_dataset import AstGraphDataset


if __name__ == '__main__':
    set_device(args.gpu)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    prog_dict = ProgDict(args.data_dir)

    model = GnnMulticlass(args, prog_dict, has_anchor=False).to(args.device)

    eval_func = partial(multiclass_eval_dataset, fn_parse_eval_nn_args=gnn_eval_nn_args)
    if args.phase == 'eval': 
        db_eval = AstGraphDataset(args, prog_dict, args.data_dir, 'eval')
        eval_loader = db_eval.get_test_loader(args)
        assert args.model_dump is not None
        model_dump = os.path.join(args.save_dir, args.model_dump)
        print('loading model from', model_dump)
        model.load_state_dict(torch.load(model_dump, map_location=args.device))    
        eval_func(model, 'eval', eval_loader)
        sys.exit()

    db_dev = AstGraphDataset(args, prog_dict, args.data_dir, 'dev')
    db_train = AstGraphDataset(args, prog_dict, args.data_dir, 'train')
    train_loop(prog_dict, model, db_train, db_dev, eval_func,
               nn_arg_constructor=gnn_arg_constructor)
