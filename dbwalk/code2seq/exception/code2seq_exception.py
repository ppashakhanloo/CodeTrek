import numpy as np
import random
import os
import sys
import torch

from dbwalk.data_util.dataset import ProgDict
from dbwalk.common.configs import cmd_args, set_device

from dbwalk.code2seq.data_util.ast_path_dataset import AstPathDataset
from dbwalk.training.train import train_loop, multiclass_eval_dataset
from dbwalk.code2seq.model import MultiClassCode2seqNet


if __name__ == '__main__':
    set_device(cmd_args.gpu)
    np.random.seed(cmd_args.seed)
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    prog_dict = ProgDict(cmd_args.data_dir)

    model = MultiClassCode2seqNet(cmd_args, prog_dict).to(cmd_args.device)
    if cmd_args.phase == 'eval': 
        db_eval = AstPathDataset(cmd_args, prog_dict, cmd_args.data_dir, 'eval')
        eval_loader = db_eval.get_test_loader(cmd_args)
        assert cmd_args.model_dump is not None
        model_dump = os.path.join(cmd_args.save_dir, cmd_args.model_dump)
        print('loading model from', model_dump)        
        model.load_state_dict(torch.load(model_dump, map_location=cmd_args.device))
        multiclass_eval_dataset(model, 'eval', eval_loader)
        sys.exit()

    db_train = AstPathDataset(cmd_args, prog_dict, cmd_args.data_dir, 'train', sample_prob=None)
    db_dev = AstPathDataset(cmd_args, prog_dict, cmd_args.data_dir, 'dev')
    train_loop(prog_dict, model, db_train, db_dev, multiclass_eval_dataset)
