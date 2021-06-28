import sys
import csv
import json
import os
from pygraphviz import Node
from data_prep.random_walk.datapoint import DataPoint, TrajNodeValue
from data_prep.random_walk.walkutils import WalkUtils
from data_prep.random_walk.randomwalk import RandomWalker
from typing import List

MAX_NUM_WALKS=100

def get_misuses(edb_dir, label, pred_kind):
    diff_len = []
    misuse_locs = []
    with open(os.path.join(edb_dir, "var_misuses.csv")) as misuse_file:
        reader = csv.reader(misuse_file, delimiter=',')
        for row in reader:
            if label == "correct":
                loc = [ x.strip() for x in row[:5] ] + [str(int(row[5])-1)]
                if loc[1][0] == "@":
                    loc[3] = str(int(loc[3]) + 1)
                if loc not in misuse_locs:
                    misuse_locs.append(loc)
                    diff_len.append(len(row[1]) - len(row[7]))
            else:
                assert label == "misuse"
                loc = [ x.strip() for x in row[6:-1] ] + [str(int(row[-1])-1)]
                if loc[1][0] == "@":
                    loc[3] = str(int(loc[3]) + 1)
                if loc not in misuse_locs:
                    misuse_locs.append(loc)
                    diff_len.append(len(row[1]) - len(row[7]))

    assert len(misuse_locs) > 0
    m_loc = misuse_locs[0]
    if diff_len[0] != 0:
        misuse_locs.append([m_loc[0], m_loc[1], m_loc[2], m_loc[3], m_loc[4], str(int(m_loc[5])+diff_len[0])])
 
    misuse_loc_ids = []
    with open(os.path.join(edb_dir, "locations_ast.bqrs.csv")) as locations_ast,\
            open(os.path.join(edb_dir, "py_locations.bqrs.csv")) as py_locations:
        ast_loc_reader = csv.reader(locations_ast, delimiter=',')
        ast_loc_rows = list(ast_loc_reader)
        ast_loc_ids = []
        for loc in misuse_locs:
            for row in ast_loc_rows:
                if loc[2:] == row[2:]:
                    ast_loc_ids.append(row)
        py_loc_reader = csv.reader(py_locations, delimiter=',')
        py_loc_rows = list(py_loc_reader)
        for loc in ast_loc_ids:
            for row in py_loc_rows:
                if loc[0] == row[0]:
                    misuse_loc_ids.append(row)
    assert len(misuse_loc_ids) > 0

    v_misuses = set()
    with open(os.path.join(edb_dir, "py_exprs.bqrs.csv")) as py_variables:
        reader = csv.reader(py_variables, delimiter=',')
        for iter, row in enumerate(reader):
            for id in misuse_loc_ids:
                if row[0] == id[1]:
                    v_misuses.add('py_exprs(' + ','.join(row) + ')')
    assert len(v_misuses) > 0
    return v_misuses

def prep_walks(edb_path, gt, pred_kind, graph, walks_or_graphs):
  anchor = list(get_misuses(edb_path, gt, pred_kind)).pop()
  anchor_node = None
  for node in graph.nodes():
    element = graph.nodes[node]['label']
    if element == anchor:
      anchor_node = node
  walks = []
  if not anchor_node:
    raise Exception('there is no anchor node :(')
  anchor_label = graph.nodes[anchor_node]['label']
  if walks_or_graphs == 'walks':
    random_walker = RandomWalker(graph, anchor_label, 'python')
    walks = random_walker.random_walk(max_num_walks=MAX_NUM_WALKS, min_num_steps=4, max_num_steps=24)
  return walks, TrajNodeValue(anchor_label)

def main(args):
  gv_file = args[1]
  edb_path = args[2]
  gt = args[3]
  out_file = args[4]
  walks_or_graphs = args[5]
  pred_kind = args[6]
  assert pred_kind in ['prog_cls', 'loc_cls', 'loc_rep']

  var_access_names = []
  with open(os.path.join(edb_path, "var_accesses.bqrs.csv"), 'r') as va, \
       open(os.path.join(edb_path, "py_exprs.bqrs.csv")) as ne:
    va_lines = [l.strip() for l in va.readlines()[1:]]
    ne_lines = [l.strip() for l in ne.readlines()[1:]]
    for v in va_lines:
      for n in ne_lines:
        if v.split(',')[0] == n.split(',')[0]:
          var_access_names.append('py_exprs(' + n + ')')
  graph = RandomWalker.load_graph_from_gv(gv_file)
  walklist = []
  if pred_kind == 'prog_cls':
    anchors = var_access_names
    anchor_nodes = set()
    for node in graph.nodes():
      element = graph.nodes[node]['label']
      if element in anchors:
        anchor_nodes.add(node)
    walks = []
    traj_anchors = []
    for anchor in anchor_nodes:
      anchor_label = graph.nodes[anchor]['label']
      if walks_or_graphs == 'walks':
        random_walker = RandomWalker(graph, anchor_label, 'python')
        walks += random_walker.random_walk(max_num_walks=MAX_NUM_WALKS//len(anchor_nodes),\
                                         min_num_steps=4, max_num_steps=24)
      traj_anchors.append(TrajNodeValue(anchor_label))
    trajectories = [WalkUtils.build_trajectory(walk) for walk in RandomWalker.padding(walks, MAX_NUM_WALKS)]
    walklist.append(DataPoint(traj_anchors, trajectories, [], gt, gv_file).to_dict())
  elif pred_kind == 'loc_cls':
    walks, traj_anchor = prep_walks(edb_path, gt, pred_kind, graph, walks_or_graphs)
    trajectories = [WalkUtils.build_trajectory(walk) for walk in walks]
    walklist.append(DataPoint([traj_anchor], trajectories, [], gt, gv_file).to_dict())
  elif pred_kind == 'loc_rep':
    # traj_anchor: error_location (correct/misuse)
    walks, traj_anchor = prep_walks(edb_path, gt, pred_kind, graph, walks_or_graphs)
    candidates = var_access_names # all var accesses, repair candidates
    anchor_nodes = set()
    for node in graph.nodes():
      element = graph.nodes[node]['label']
      if element in candidates:
        anchor_nodes.add(node)
    #####... complete loc_rep.
    targets = [] # all correct var accesses
    #
  else:
    raise NotImplementedError(pred_kind)

  with open(out_file, 'w') as f:
    f.write(json.dumps(walklist))

if __name__ == '__main__':
    main(sys.argv)
