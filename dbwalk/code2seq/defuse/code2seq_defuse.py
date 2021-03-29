from __future__ import print_function, division

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import random
import torch.optim as optim

from dbwalk.data_util.dataset import ProgDict
from dbwalk.common.configs import cmd_args, set_device

from dbwalk.code2seq.data_util.ast_path_dataset import AstPathDataset
from dbwalk.code2seq.model import BinaryCode2seqNet
from dbwalk.training.train import train_loop
from dbwalk.var_misuse.main import eval_dataset
from sklearn.metrics import roc_auc_score


if __name__ == '__main__':
    set_device(cmd_args.gpu)
    np.random.seed(cmd_args.seed)
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    prog_dict = ProgDict(cmd_args.data_dir)

    model = BinaryCode2seqNet(cmd_args, prog_dict).to(cmd_args.device)
    if cmd_args.phase == 'test': 
        db_eval = AstPathDataset(cmd_args, prog_dict, cmd_args.data_dir, 'eval')
        eval_loader = db_eval.get_test_loader(cmd_args)
        assert cmd_args.model_dump is not None
        model_dump = os.path.join(cmd_args.save_dir, cmd_args.model_dump)
        print('loading model from', model_dump)        
        model.load_state_dict(torch.load(model_dump, map_location=cmd_args.device))
        eval_dataset(model, 'eval', eval_loader)
        sys.exit()

    db_train = AstPathDataset(cmd_args, prog_dict, cmd_args.data_dir, 
                              sample_prob=None,#{'used': 0.5, 'unused': 0.5}, 
                              phase='train')
    db_dev = AstPathDataset(cmd_args, prog_dict, cmd_args.data_dir, 'dev')
    train_loop(prog_dict, model, db_train, db_dev, eval_dataset)
