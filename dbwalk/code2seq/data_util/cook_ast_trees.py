import json
import os
from tqdm import tqdm
import networkx as nx
import pickle as cp
from dbwalk.common.configs import cmd_args
from dbwalk.common.consts import TOK_PAD, var_idx2name, UNK
from dbwalk.data_util.graph_holder import GraphHolder
from dbwalk.data_util.cook_data import get_or_add
from data_prep.tokenizer import tokenizer
from dbwalk.ggnn.data_util.cook_ast_graphs import main_cook_ast


if __name__ == '__main__':
    main_cook_ast(etype_white_list=['Child'], edge_unk=False)
