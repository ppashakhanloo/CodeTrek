from __future__ import print_function, division

import os
import sys
import torch
import numpy as np
import random
import torch.optim as optim
from torch.utils.data import DataLoader
from dbwalk.data_util.dataset import InMemDataest, PreGeneratedWalkDataset, ProgDict, FastOnlineWalkDataset
from dbwalk.common.configs import cmd_args, get_torch_device
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from dbwalk.model.classifier import BinaryNet
from dbwalk.training.train import train_loop, binary_eval_dataset, train_mp
import torch.multiprocessing as mp


if __name__ == '__main__':
    np.random.seed(cmd_args.seed)
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    prog_dict = ProgDict(cmd_args.data_dir)
    model = BinaryNet(cmd_args, prog_dict)

    if cmd_args.online_walk_gen:
        db_class = FastOnlineWalkDataset
    else:
        db_class = InMemDataest

    if cmd_args.phase == 'eval':
        assert cmd_args.model_dump is not None
        model_dump = os.path.join(cmd_args.save_dir, cmd_args.model_dump)
        print('loading model from', model_dump)
        device = get_torch_device(cmd_args.gpu)
        model = model.to(device)
        model.load_state_dict(torch.load(model_dump))
        db_eval = db_class(cmd_args, prog_dict, cmd_args.data_dir, 'eval')
        eval_loader = db_eval.get_test_loader(cmd_args)
        binary_eval_dataset(model, cmd_args.phase, eval_loader, device)
        sys.exit()

    if cmd_args.phase == 'single_test':
        assert cmd_args.model_dump is not None
        print('testing one source...')
        model_dump = os.path.join(cmd_args.save_dir, cmd_args.model_dump)
        print('loading model from', model_dump)
        device = get_torch_device(cmd_args.gpu)
        model = model.to(device)
        model.load_state_dict(torch.load(model_dump))
        db_eval = db_class(cmd_args, prog_dict, cmd_args.data_dir, 'eval')
        eval_loader = db_eval.get_test_loader(cmd_args)
        binary_eval_dataset(model, cmd_args.phase, eval_loader, device)
        sys.exit()

    db_train = db_class(cmd_args, prog_dict, cmd_args.data_dir, 'train', 
                        sample_prob={'used': 0.5, 'unused': 0.5},
                        shuffle_var=cmd_args.shuffle_var)
    db_dev = db_class(cmd_args, prog_dict, cmd_args.data_dir, 'dev')

    if cmd_args.num_train_proc > 1:
        mp.set_start_method('spawn')
        if cmd_args.gpu_list is not None:
            devices = [get_torch_device(int(x.strip())) for x in cmd_args.gpu_list.split(',')]
        else:
            devices = ['cpu'] * cmd_args.num_train_proc
        assert len(devices) == cmd_args.num_train_proc
        procs = []
        for rank, device in enumerate(devices):
            proc = mp.Process(target=train_mp,
                    cmd_args=(cmd_args, rank, device, prog_dict, model, db_train, db_dev, binary_eval_dataset))
            proc.start()
        for proc in procs:
            proc.join()
    else:
        torch.cuda.empty_cache()
        train_loop(cmd_args, get_torch_device(cmd_args.gpu), prog_dict, model, db_train, db_dev, binary_eval_dataset)
