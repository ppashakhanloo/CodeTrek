import sys
import csv
import json
import os
from pygraphviz import Node
from data_prep.random_walk.datapoint import DataPoint, TrajNodeValue
from data_prep.random_walk.walkutils import WalkUtils
from data_prep.random_walk.randomwalk import RandomWalker
from typing import List
import random

from data_prep.utils.gcp_utils import gcp_copy_from, gcp_copy_to

MAX_NUM_WALKS = 200

def main(args: List[str]) -> None:
  gv_file = args[1]
  edb_path = args[2]
  out_file = args[3]
  walks_or_graphs = args[4]
  pred_kind = args[5]

  anchor_nodes = []
  with open(os.path.join(edb_path, 'shadow_global.bqrs.csv'), 'r') as f:
    lines = f.readlines()
    if len(lines) == 1:
      label = 'bright'
    else:
      label = 'shadowed'
    var_names = set()
    for line in lines:
      line = line.strip().split(',')
      name = line[1].replace("Local variable '", '').replace("' shadows a global variable defined $@.", '')
      var_names.add(name[1:-1])

  variables = []
  with open(os.path.join(edb_path, 'variable.bqrs.csv')) as f:
    lines = f.readlines()
    for line in lines[1:]:
      line = line.strip().split(',')
      variables.append('variable(' + line[0] + ',' + line[1] + ',' + line[2][1:-1] + ')')
      if line[2][1:-1] in var_names:
          anchor_nodes.append('variable(' + line[0] + ',' + line[1] + ',' + line[2][1:-1] + ')')


  if len(anchor_nodes) == 0:
    anchor_nodes = random.choices(variables, k=4)
  if len(anchor_nodes) > 10:
    anchor_nodes = anchor_nodes[:10]

  graph = RandomWalker.load_graph_from_gv(gv_file)

  walklist = []
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
  data_point = DataPoint(traj_anchors, trajectories, [], label, gv_file)
  walklist.append(data_point.to_dict())

  with open(out_file, 'w') as op_file:
    op_file.write(json.dumps(walklist))

if __name__ == '__main__':
  main(sys.argv)
