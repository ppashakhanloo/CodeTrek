import numpy as np
import random
import os
import sys
import torch

from dbwalk.data_util.dataset import ProgDict
from dbwalk.common.configs import args, set_device, get_torch_device

from dbwalk.code2seq.data_util.ast_path_dataset import AstPathDataset
from dbwalk.training.train import train_loop, multiclass_eval_dataset
from dbwalk.code2seq.model import MultiClassCode2seqNet


if __name__ == '__main__':
    set_device(args.gpu)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    prog_dict = ProgDict(args.data_dir)

    model = MultiClassCode2seqNet(args, prog_dict).to(args.device)
    if args.phase == 'eval':
        db_eval = AstPathDataset(args, prog_dict, args.data_dir, 'eval')
        eval_loader = db_eval.get_test_loader(args)
        assert args.model_dump is not None
        model_dump = os.path.join(args.save_dir, args.model_dump)
        print('loading model from', model_dump)
        model.load_state_dict(torch.load(model_dump, map_location=args.device))
        multiclass_eval_dataset(model, 'eval', eval_loader)
        sys.exit()

    db_train = AstPathDataset(args, prog_dict, args.data_dir, 'train', sample_prob=None)
    db_dev = AstPathDataset(args, prog_dict, args.data_dir, 'dev')
    train_loop(args, get_torch_device(args.gpu), prog_dict, model, db_train,
               db_dev, multiclass_eval_dataset)
