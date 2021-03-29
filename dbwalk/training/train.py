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


def path_based_arg_constructor(nn_args):
    node_idx, edge_idx, node_val_mat, label = nn_args
    if node_val_mat is not None:
        node_val_mat = torch.sparse_coo_tensor(*node_val_mat).to(cmd_args.device)
    if edge_idx is not None:
        edge_idx = edge_idx.to(cmd_args.device)
    nn_args = {'node_idx': node_idx.to(cmd_args.device),
               'edge_idx': edge_idx,
               'node_val_mat': node_val_mat,
               'label': label.to(cmd_args.device)}
    return nn_args


def eval_path_based_nn_args(nn_args):
    node_idx, edge_idx, node_val_mat, label = nn_args
    if node_idx is None:
        return None, None
    if node_val_mat is not None:
        node_val_mat = torch.sparse_coo_tensor(*node_val_mat).to(cmd_args.device)
    if edge_idx is not None:
        edge_idx = edge_idx.to(cmd_args.device)
    nn_args = {'node_idx': node_idx.to(cmd_args.device),
               'edge_idx': edge_idx,
               'node_val_mat': node_val_mat}
    return nn_args, label


def multiclass_eval_dataset(model, phase, eval_loader, fn_parse_eval_nn_args=eval_path_based_nn_args):
    true_labels = []
    pred_labels = []
    model.eval()
    pbar = tqdm(eval_loader)
    for nn_args in pbar:
        with torch.no_grad():
            nn_args, label = fn_parse_eval_nn_args(nn_args)
            if nn_args is None:
                continue
            logits = model(**nn_args)
            pred_labels += torch.argmax(logits, dim=1).data.cpu().numpy().flatten().tolist()
            true_labels += label.data.numpy().flatten().tolist()
        pbar.set_description('evaluating %s' % phase)
    pred_labels = np.array(pred_labels, dtype=np.int32)
    acc = np.mean(pred_labels == np.array(true_labels, dtype=np.int32))
    print('%s acc: %.4f' % (phase, acc))
    return acc


def binary_eval_dataset(model, phase, eval_loader, fn_parse_eval_nn_args=eval_path_based_nn_args):
    true_labels = []
    pred_probs = []
    model.eval()
    pbar = tqdm(eval_loader)
    for nn_args in pbar:
        with torch.no_grad():
            nn_args, label = fn_parse_eval_nn_args(nn_args)
            if nn_args is None:
                continue
            pred = model(**nn_args).data.cpu().numpy()
            pred_probs += pred.flatten().tolist()
            true_labels += label.data.numpy().flatten().tolist()
        pbar.set_description('evaluating %s' % phase)
    roc_auc = roc_auc_score(true_labels, pred_probs)
    pred_label = np.where(np.array(pred_probs) > 0.5, 1, 0)
    acc = np.mean(pred_label == np.array(true_labels, dtype=pred_label.dtype))
    print('%s auc: %.4f, acc: %.4f' % (phase, roc_auc, acc))
    return roc_auc


def train_loop(prog_dict, model, db_train,
               db_dev=None, 
               fn_eval=None,
               nn_arg_constructor=path_based_arg_constructor):
    train_loader = db_train.get_train_loader(cmd_args)
    dev_loader = db_dev.get_test_loader(cmd_args)
    optimizer = optim.Adam(model.parameters(), lr=cmd_args.learning_rate)
    train_iter = iter(train_loader)

    best_metric = -1
    for epoch in range(cmd_args.num_epochs):
        pbar = tqdm(range(cmd_args.iter_per_epoch))
        model.train()
        for i in pbar:
            try:
                nn_args = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                nn_args = next(train_iter)
            optimizer.zero_grad()
            nn_args = nn_arg_constructor(nn_args)
            loss = model(**nn_args)
            loss.backward()
            if cmd_args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cmd_args.grad_clip)
            pbar.set_description('step %d, loss: %.2f' % (cmd_args.iter_per_epoch * epoch + i, loss.item()))
            optimizer.step()
        if fn_eval is not None:
            auc = fn_eval(model, 'dev', dev_loader)
            if auc > best_metric:
                best_metric = auc
                print('saving model with best dev metric: %.4f' % best_metric)
                torch.save(model.state_dict(), os.path.join(cmd_args.save_dir, 'model-best_dev.ckpt'))
