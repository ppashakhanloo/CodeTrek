from __future__ import print_function, division

import os
import sys
import torch
import numpy as np
import random
import torch.optim as optim
from torch.utils.data import DataLoader
from dbwalk.data_util.dataset import InMemDataest, ProgDict, FastOnlineWalkDataset, SlowOnlineWalkDataset
from dbwalk.common.configs import cmd_args, set_device
from tqdm import tqdm
from dbwalk.model.classifier import MulticlassNet
from dbwalk.training.train import train_loop


def eval_dataset(model, phase, eval_loader):
    true_labels = []
    pred_labels = []
    model.eval()
    pbar = tqdm(eval_loader)
    for node_idx, edge_idx, node_val_mat, label in pbar:
        if node_idx is None:
            continue
        with torch.no_grad():
            if node_val_mat is not None:
                node_val_mat = torch.sparse_coo_tensor(*node_val_mat).to(cmd_args.device)
            if edge_idx is not None:
                edge_idx = edge_idx.to(cmd_args.device)
            logits = model(node_idx.to(cmd_args.device), edge_idx, node_val_mat=node_val_mat)
            pred_labels += torch.argmax(logits, dim=1).data.cpu().numpy().flatten().tolist()
            true_labels += label.data.numpy().flatten().tolist()
        pbar.set_description('evaluating %s' % phase)
    pred_labels = np.array(pred_labels, dtype=np.int32)
    acc = np.mean(pred_labels == np.array(true_labels, dtype=np.int32))
    print('%s acc: %.4f' % (phase, acc))
    return acc


if __name__ == '__main__':
    set_device(cmd_args.gpu)
    np.random.seed(cmd_args.seed)
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    prog_dict = ProgDict(cmd_args.data_dir)
    model = MulticlassNet(cmd_args, prog_dict).to(cmd_args.device)

    if cmd_args.online_walk_gen:
        db_class = FastOnlineWalkDataset
    else:
        db_class = InMemDataest

    if cmd_args.phase != 'train':
        assert cmd_args.model_dump is not None
        model_dump = os.path.join(cmd_args.save_dir, cmd_args.model_dump)
        print('loading model from', model_dump)
        model.load_state_dict(torch.load(model_dump, map_location=cmd_args.device))
        db_eval = db_class(cmd_args, prog_dict, cmd_args.data_dir, cmd_args.phase)
        eval_loader = db_eval.get_test_loader(cmd_args)
        eval_dataset(model, cmd_args.phase, eval_loader)
        sys.exit()

    db_train = db_class(cmd_args, prog_dict, cmd_args.data_dir, 'train', sample_prob=None, shuffle_var=cmd_args.shuffle_var)
    db_dev = db_class(cmd_args, prog_dict, cmd_args.data_dir, 'dev')
    train_loop(prog_dict, model, db_train, db_dev, eval_dataset)
