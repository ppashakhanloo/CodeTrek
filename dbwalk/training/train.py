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



def train_loop(prog_dict, model, db_train, db_dev=None, fn_eval=None):
    train_loader = DataLoader(db_train, batch_size=cmd_args.batch_size, shuffle=True, drop_last=True, 
                              collate_fn=db_train.collate_fn, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=cmd_args.learning_rate)
    train_iter = iter(train_loader)

    best_metric = -1
    for epoch in range(cmd_args.num_epochs):
        pbar = tqdm(range(cmd_args.iter_per_epoch))
        model.train()
        for i in pbar:
            try:
                node_idx, edge_idx, label = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                node_idx, edge_idx, label = next(train_iter)
            optimizer.zero_grad()
            loss = model(node_idx.to(cmd_args.device), edge_idx.to(cmd_args.device), label.to(cmd_args.device))
            loss.backward()
            if cmd_args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cmd_args.grad_clip)
            pbar.set_description('step %d, loss: %.2f' % (cmd_args.iter_per_epoch * epoch + i, loss.item()))
            optimizer.step()
        if fn_eval is not None:
            auc = fn_eval(model, db_dev)
            if auc > best_metric:
                best_metric = auc
                print('saving model with best dev metric: %.4f' % best_metric)
                torch.save(model.state_dict(), os.path.join(cmd_args.save_dir, 'model-best_dev.ckpt'))