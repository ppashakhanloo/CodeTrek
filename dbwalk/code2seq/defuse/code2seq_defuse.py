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
from dbwalk.common.configs import cmd_args, get_torch_device

from dbwalk.code2seq.data_util.ast_path_dataset import AstPathDataset
from dbwalk.code2seq.model import BinaryCode2seqNet
from dbwalk.training.train import train_loop, binary_eval_dataset


if __name__ == '__main__':
    np.random.seed(cmd_args.seed)
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    prog_dict = ProgDict(cmd_args.data_dir)

    model = BinaryCode2seqNet(cmd_args, prog_dict)
    if cmd_args.phase == 'eval':
        db_eval = AstPathDataset(cmd_args, prog_dict, cmd_args.data_dir, 'eval')
        eval_loader = db_eval.get_test_loader(cmd_args)
        assert cmd_args.model_dump is not None
        model_dump = os.path.join(cmd_args.save_dir, cmd_args.model_dump)
        print('loading model from', model_dump)
        model.load_state_dict(torch.load(model_dump, map_location=get_torch_device(cmd_args.gpu)))
        binary_eval_dataset(model, 'eval', eval_loader, get_torch_device(cmd_args.gpu))
        sys.exit()

    db_train = AstPathDataset(cmd_args, prog_dict, cmd_args.data_dir,
                              sample_prob=None,#{'used': 0.5, 'unused': 0.5},
                              phase='train')
    db_dev = AstPathDataset(cmd_args, prog_dict, cmd_args.data_dir, 'dev')
    train_loop(cmd_args, get_torch_device(cmd_args.gpu), prog_dict, model, db_train, db_dev, binary_eval_dataset)
