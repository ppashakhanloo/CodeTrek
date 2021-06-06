from __future__ import print_function, division

import os
import sys
import torch
import numpy as np
import random
import torch.optim as optim
from torch.utils.data import DataLoader
from dbwalk.data_util.dataset import InMemDataest, ProgDict, FastOnlineWalkDataset
from dbwalk.common.configs import args, get_torch_device
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from dbwalk.model.classifier import BinaryNet
from dbwalk.training.train import train_loop, binary_eval_dataset, train_mp

import torch.multiprocessing as mp

if __name__ == '__main__':
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    prog_dict = ProgDict(args.data_dir)
    model = BinaryNet(args, prog_dict)

    if args.online_walk_gen:
        db_class = FastOnlineWalkDataset
    else:
        db_class = InMemDataest

    if args.phase == 'eval':
        assert args.model_dump is not None
        model_dump = os.path.join(args.save_dir, args.model_dump)
        print('loading model from', model_dump)
        device = get_torch_device(args.gpu)
        model = model.to(device)
        model.load_state_dict(torch.load(model_dump))
        db_eval = db_class(args, prog_dict, args.data_dir, 'eval')
        eval_loader = db_eval.get_test_loader(args)
        binary_eval_dataset(model, args.phase, eval_loader, device)
        sys.exit()
    torch.cuda.empty_cache()
    db_train = db_class(args, prog_dict, args.data_dir, 'train', 
                        sample_prob={'used': 0.5, 'unused': 0.5},
                        shuffle_var=args.shuffle_var)
    db_dev = db_class(args, prog_dict, args.data_dir, 'dev')

    if args.num_train_proc > 1:
        mp.set_start_method('spawn')
        if args.gpu_list is not None:
            devices = [get_torch_device(int(x.strip())) for x in args.gpu_list.split(',')]
        else:
            devices = ['cpu'] * args.num_train_proc
        assert len(devices) == args.num_train_proc
        procs = []
        for rank, device in enumerate(devices):
            torch.cuda.empty_cache()
            proc = mp.Process(target=train_mp,
                    args=(args, rank, device, prog_dict, model, db_train, db_dev, binary_eval_dataset))
            proc.start()
        for proc in procs:
            proc.join()
    else:
        torch.cuda.empty_cache()
        train_loop(args, get_torch_device(args.gpu), prog_dict, model, db_train, db_dev, binary_eval_dataset)
