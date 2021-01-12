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


def eval_dataset(model, eval_set):
    eval_loader = DataLoader(eval_set, batch_size=cmd_args.batch_size, shuffle=False, drop_last=False, collate_fn=eval_set.collate_fn, num_workers=0)
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
        pbar.set_description('evaluating %s' % eval_set.phase)
    roc_auc = roc_auc_score(true_labels, pred_probs)
    pred_label = np.where(np.array(pred_probs) > 0.5, 1, 0)
    acc = np.mean(pred_label == np.array(true_labels, dtype=pred_label.dtype))
    print('%s auc: %.4f, acc: %.4f' % (eval_set.phase, roc_auc, acc))
    return roc_auc


if __name__ == '__main__':
    set_device(cmd_args.gpu)
    np.random.seed(cmd_args.seed)
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    prog_dict = ProgDict(cmd_args.data_dir)

    model = BinaryNet(cmd_args, prog_dict).to(cmd_args.device)

    if cmd_args.phase == 'test':        
        assert cmd_args.model_dump is not None
        model_dump = os.path.join(cmd_args.save_dir, cmd_args.model_dump)
        print('loading model from', model_dump)
        model.load_state_dict(torch.load(model_dump, map_location=cmd_args.device))
        db_test = InMemDataest(prog_dict, cmd_args.data_dir, 'test')
        eval_dataset(model, db_test)
        sys.exit()

    db_train = InMemDataest(prog_dict, cmd_args.data_dir, 'train', sample_prob=[0.5, 0.5])

    db_dev = InMemDataest(prog_dict, cmd_args.data_dir, 'dev')
    train_loader = DataLoader(db_train, batch_size=cmd_args.batch_size, shuffle=True, drop_last=True, 
                              collate_fn=db_train.collate_fn, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=cmd_args.learning_rate)
    train_iter = iter(train_loader)

    best_auc = -1
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
        auc = eval_dataset(model, db_dev)
        if auc > best_auc:
            best_auc = auc
            print('saving model with best dev auc')
            torch.save(model.state_dict(), os.path.join(cmd_args.save_dir, 'model-best_dev.ckpt'))
