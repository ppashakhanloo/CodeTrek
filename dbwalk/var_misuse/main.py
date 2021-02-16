from __future__ import print_function, division

import os
import sys
import torch
import numpy as np
import random
import torch.optim as optim
from torch.utils.data import DataLoader
from dbwalk.data_util.dataset import InMemDataest, ProgDict
from dbwalk.common.configs import cmd_args, set_device
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from dbwalk.model.classifier import BinaryNet
from dbwalk.training.train import train_loop


def eval_dataset(model, phase, eval_loader):
    true_labels = []
    pred_probs = []
    model.eval()
    pbar = tqdm(eval_loader)
    for node_idx, edge_idx, label in pbar:
        if node_idx is None:
            continue
        with torch.no_grad():
            pred = model(node_idx.to(cmd_args.device), edge_idx.to(cmd_args.device)).data.cpu().numpy()
            pred_probs += pred.flatten().tolist()
            true_labels += label.data.numpy().flatten().tolist()
        pbar.set_description('evaluating %s' % phase)
    roc_auc = roc_auc_score(true_labels, pred_probs)
    pred_label = np.where(np.array(pred_probs) > 0.5, 1, 0)
    acc = np.mean(pred_label == np.array(true_labels, dtype=pred_label.dtype))
    print('%s auc: %.4f, acc: %.4f' % (phase, roc_auc, acc))
    return roc_auc


if __name__ == '__main__':
    set_device(cmd_args.gpu)
    np.random.seed(cmd_args.seed)
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    prog_dict = ProgDict(cmd_args.data_dir)
    model = BinaryNet(cmd_args, prog_dict).to(cmd_args.device)
    db_class = InMemDataest

    if cmd_args.phase == 'test':        
        assert cmd_args.model_dump is not None
        model_dump = os.path.join(cmd_args.save_dir, cmd_args.model_dump)
        print('loading model from', model_dump)
        model.load_state_dict(torch.load(model_dump, map_location=cmd_args.device))
        db_eval = db_class(cmd_args, prog_dict, cmd_args.data_dir, 'eval')
        eval_loader = db_eval.get_test_loader(cmd_args)
        eval_dataset(model, 'eval', eval_loader)
        sys.exit()

    db_train = db_class(cmd_args, prog_dict, cmd_args.data_dir, 'train', sample_prob=[0.5, 0.5], shuffle_var=cmd_args.shuffle_var)
    db_dev = db_class(cmd_args, prog_dict, cmd_args.data_dir, 'dev')
    train_loop(prog_dict, model, db_train, db_dev, eval_dataset)
