import sys
import csv
import json
import os
from pygraphviz import Node
from data_prep.random_walk.datapoint import DataPoint, TrajNodeValue
from data_prep.random_walk.walkutils import WalkUtils
from data_prep.random_walk.randomwalk import RandomWalker
from typing import List

from data_prep.utils.gcp_utils import gcp_copy_from, gcp_copy_to

MAX_NUM_WALKS = 100

def get_semmle_defs(edb_path):
  name_exprs = []
  with open(os.path.join(edb_path, "py_exprs.bqrs.csv"), 'r') as f:
    reader = csv.reader(f, delimiter=',')
    rows = list(reader)
    for row in rows[1:]:
      if row[1] == '18':
        name_exprs.append(row)

  ctx_stores = []
  with open(os.path.join(edb_path, "py_expr_contexts.bqrs.csv"), 'r') as f:
    reader = csv.reader(f, delimiter=',')
    rows = list(reader)
    for row in rows[1:]:
      if row[1] in ['1', '4', '5']:
        ctx_stores.append(row)

  defs = []
  for e in name_exprs:
    for ctx in ctx_stores:
      if e[0] == ctx[2]:
        defs.append(e)

  # defs are py_exprs(..)
  return defs

def get_semmle_unused_vars(edb_path):
  with open(os.path.join(edb_path, "unused_var.bqrs.csv"), 'r') as f:
    rows = list(csv.reader(f, delimiter=','))
    unused_vars = [row for row in rows[1:]] # expr_id, var_id
  return unused_vars

def get_semmle_locs(edb_path):
  ast_locations, py_locations = [], []
  with open(os.path.join(edb_path, "locations_ast.bqrs.csv"), 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
      ast_locations.append(line.strip().split(','))
  with open(os.path.join(edb_path, "py_locations.bqrs.csv"), 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
      py_locations.append(line.strip().split(','))
  return ast_locations, py_locations

def main(args: List[str]) -> None:
  gv_file = args[1]
  edb_path = args[2]
  out_file = args[3]
  walks_or_graphs = args[4]
  pred_kind = args[5]

  unused_vars = get_semmle_unused_vars(edb_path)
  defs = get_semmle_defs(edb_path)

  anchor_nodes = []
  if pred_kind == 'prog_cls':
    if len(unused_vars) > 0:
      end_label = 'unused'
    else:
      end_label = 'used'
    for n in name_exprs:
      for ctx in ctx_stores:
        if n[0] == ctx[2]:
          anchor_nodes.append('py_exprs(' + ','.join(n) + ')')
  elif pred_kind == 'loc_cls':
    for n in name_exprs:
      for ctx in ctx_stores:
        if n[0] == ctx[2]:
          anch = 'py_exprs(' + ','.join(n) + ')'
          if n[0] in unused_vars:
            anchor_nodes.append((anch, 'unused'))
          else:
            anchor_nodes.append((anch, 'used'))
  else:
    raise NotImplementedError(pred_kind)

  graph = RandomWalker.load_graph_from_gv(gv_file)

  walklist = []
  if pred_kind == 'prog_cls':
    walks_all = []
    traj_anchors = []
    for anchor in anchor_nodes:
      walks = []
      if walks_or_graphs == 'walks':
        random_walk = RandomWalker(graph, anchor, 'python')
        walks = random_walk.random_walk(max_num_walks=MAX_NUM_WALKS//len(anchor_nodes), min_num_steps=4, max_num_steps=24)
      traj_anchors.append(TrajNodeValue(anchor))
      walks_all += walks
    if walks_or_graphs == 'walks':
      trajectories = [WalkUtils.build_trajectory(walk) for walk in RandomWalker.padding(walks_all, MAX_NUM_WALKS)]
    else:
      trajectories = []
    data_point = DataPoint(traj_anchors, trajectories, [], end_label, gv_file)
    walklist.append(data_point.to_dict())
  elif pred_kind == 'loc_cls':
    for anchor, label in anchor_nodes:
      walks = []
      traj_anchors = [TrajNodeValue(anchor)]
      if walks_or_graphs == 'walks':
        random_walk = RandomWalker(graph, anchor, 'python')
        walks = random_walk.random_walk(max_num_walks=MAX_NUM_WALKS, min_num_steps=4, max_num_steps=24)
      trajectories = [WalkUtils.build_trajectory(walk) for walk in walks]
      data_point = DataPoint(traj_anchors, trajectories, [], label, gv_file)
      walklist.append(data_point.to_dict())
  else:
    NotImplementedError(pred_kind)

  with open(out_file, 'w') as op_file:
    op_file.write(json.dumps(walklist))


if __name__ == '__main__':
  main(sys.argv)
