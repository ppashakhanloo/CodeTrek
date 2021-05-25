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
from dbwalk.common.configs import args, get_torch_device

from dbwalk.code2seq.data_util.ast_path_dataset import AstPathDataset
from dbwalk.code2seq.model import BinaryCode2seqNet
from dbwalk.training.train import train_loop, binary_eval_dataset

if __name__ == '__main__':
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    prog_dict = ProgDict(args.data_dir)

    model = BinaryCode2seqNet(args, prog_dict)
    if args.phase == 'eval':
        db_eval = AstPathDataset(args, prog_dict, args.data_dir, 'eval')
        eval_loader = db_eval.get_test_loader(args)
        assert args.model_dump is not None
        model_dump = os.path.join(args.save_dir, args.model_dump)
        print('loading model from', model_dump)
        model.load_state_dict(torch.load(model_dump, map_location=get_torch_device(args.gpu)))
        binary_eval_dataset(model, 'eval', eval_loader, get_torch_device(args.gpu))
        sys.exit()

    db_train = AstPathDataset(args, prog_dict, args.data_dir, 'train')
    db_dev = AstPathDataset(args, prog_dict, args.data_dir, 'dev')
    train_loop(args, get_torch_device(args.gpu), prog_dict, model, db_train, db_dev, binary_eval_dataset)
