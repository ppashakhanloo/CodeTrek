from __future__ import print_function, division

import os
import sys
import torch
import numpy as np
import random
import torch.optim as optim
from torch.utils.data import DataLoader
from dbwalk.data_util.dataset import InMemDataest, ProgDict
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from functools import wraps
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
from _thread import start_new_thread
import torch.distributed as dist
import traceback


def thread_wrapped_func(func):
    """Wrapped func for torch.multiprocessing.Process.
    With this wrapper we can use OMP threads in subprocesses
    otherwise, OMP_NUM_THREADS=1 is mandatory.
    How to use:
    @thread_wrapped_func
    def func_to_wrap(args ...):
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = Queue()
        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)
    return decorated_function


def path_based_arg_constructor(nn_args, device):
    node_idx, edge_idx, node_val_mat, label = nn_args
    if node_val_mat is not None:
        node_val_mat = torch.sparse_coo_tensor(*node_val_mat).to(device)
    if edge_idx is not None:
        edge_idx = edge_idx.to(device)
    nn_args = {'node_idx': node_idx.to(device),
               'edge_idx': edge_idx,
               'node_val_mat': node_val_mat,
               'label': label.to(device)}
    return nn_args


def eval_path_based_nn_args(nn_args, device):
    node_idx, edge_idx, node_val_mat, label = nn_args
    if node_idx is None:
        return None, None
    if node_val_mat is not None:
        node_val_mat = torch.sparse_coo_tensor(*node_val_mat).to(device)
    if edge_idx is not None:
        edge_idx = edge_idx.to(device)
    nn_args = {'node_idx': node_idx.to(device),
               'edge_idx': edge_idx,
               'node_val_mat': node_val_mat}
    return nn_args, label


def multiclass_eval_dataset(model, phase, eval_loader, device, fn_parse_eval_nn_args=eval_path_based_nn_args):
    true_labels = []
    pred_labels = []
    model.eval()
    pbar = tqdm(eval_loader)
    for nn_args in pbar:
        with torch.no_grad():
            nn_args, label = fn_parse_eval_nn_args(nn_args, device)
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


def binary_eval_dataset(model, phase, eval_loader, device, fn_parse_eval_nn_args=eval_path_based_nn_args):
    true_labels = []
    pred_probs = []
    model.eval()
    pbar = tqdm(eval_loader)
    for nn_args in pbar:
        with torch.no_grad():
            nn_args, label = fn_parse_eval_nn_args(nn_args, device)
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


def train_loop(args, device, prog_dict, model, db_train,
               db_dev=None, 
               fn_eval=None,
               nn_arg_constructor=path_based_arg_constructor):
    is_distributed = args.num_train_proc > 1
    if is_distributed:
        rank = dist.get_rank()
    else:
        rank = 0
    model = model.to(device)
    train_loader = db_train.get_train_loader(args)
    dev_loader = db_dev.get_test_loader(args)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train_iter = iter(train_loader)

    best_metric = -1
    for epoch in range(args.num_epochs):
        pbar = tqdm(range(args.iter_per_epoch)) if rank == 0 else range(args.iter_per_epoch)
        model.train()
        for i in pbar:
            try:
                nn_args = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                nn_args = next(train_iter)
            optimizer.zero_grad()
            nn_args = nn_arg_constructor(nn_args, device)
            loss = model(**nn_args) / args.num_train_proc
            loss.backward()
            if is_distributed:
                for param in model.parameters():
                    if param.grad is None:
                        param.grad = param.data.new(param.data.shape).zero_()
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            pbar.set_description('step %d, loss: %.2f' % (args.iter_per_epoch * epoch + i, loss.item()))
            optimizer.step()
        if fn_eval is not None:
            if rank == 0:
                auc = fn_eval(model, 'dev', dev_loader, device)
                if auc > best_metric:
                    best_metric = auc
                    print('saving model with best dev metric: %.4f' % best_metric)
                    torch.save(model.state_dict(), os.path.join(args.save_dir, 'model-best_dev.ckpt'))
            dist.barrier()


@thread_wrapped_func
def train_mp(args, rank, device, prog_dict, model, db_train, db_dev=None, fn_eval=None, n_arg_constructor=path_based_arg_constructor):
    if args.num_train_proc > 1:
        torch.set_num_threads(1)    
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = args.port
    if device == 'cpu':
        backend = 'gloo'
    else:
        backend = 'nccl'
    dist.init_process_group(backend, rank=rank, world_size=args.num_train_proc)
    train_loop(args, device, prog_dict, model, db_train, db_dev, fn_eval)
