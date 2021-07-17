import os
import sys
import json

import networkx as nx

from data_prep.baselines.baseline_gnn_ast.create_ast import build_child_edges, fix_node_labels

def main():
  graph, neighbors, subtrees, node_of_interest, if_branches, tok1, tok2 = build_child_edges(py_file1, py_file2, task_name, pred_kind)
  graph, terminal_vars, hole_exception = fix_node_labels(graph)
  
  splits = py_file1.strip().split('/')
  filename = splits[-1]
  prog_label = splits[-2]

  # create map
  node_to_num = dict()
  for idx, node in enumerate(graph.get_nodes()):
    node_to_num[node.get_name()] = (idx + 1, node.get('label') + "(" + str(idx+1) + ")")
  
  new_graph = nx.Graph()
  for i in node_to_num:
    new_graph.add_node(node_to_num[i][0], label=node_to_num[i][1])
  for e in graph.get_edges():
    new_graph.add_edge(node_to_num[e.get_source()][0], node_to_num[e.get_destination()][0], label="child")

  new_graph = nx.nx_agraph.to_agraph(new_graph)
  new_graph = str(new_graph).replace('strict graph ""', 'graph').replace('node [label="\\N"];','')

  # save graph file
  with open('graph_' + filename + '.gv', 'w') as f:
    f.write(new_graph)

  # find var defs and uses
  defs, uses = [], []
  for v in terminal_vars:
    if v[2] == 'write':
      defs.append(v)
    if v[2] == 'read':
      uses.append(v)

  if task_name == 'varmisuse':
    if pred_kind == 'prog_cls':
      point = [{
        'anchors': [node_to_num[n[0].get_name()][1] for n in uses+defs],
        'trajectories': [],
        'hints': [],
        'label': prog_label,
        'source': py_file1
      }]
    if pred_kind == 'loc_cls':
      raise NotImplementedError(task_name + ',' + pred_kind)
    if pred_kind == 'loc_rep':
      raise NotImplementedError(task_name + ',' + pred_kind)

  if task_name == 'defuse':
    if pred_kind == 'prog_cls':
      point = [{
        'anchors': [node_to_num[d[0].get_name()][1] for d in defs],
        'trajectories': [],
        'hints': [],
        'label': prog_label,
        'source': py_file1
      }]
    if pred_kind == 'loc_cls':
      raise notImplementedError(task_name + ',' + pred_kind)

  if task_name == 'exception':
    point = [{
      'anchors': [node_to_num[hole_exception.get_name()][1]],
      'trajectories': [],
      'hints': [],
      'label': prog_label,
      'source': py_file1
    }]

  # save stub file
  with open('stub_' + filename + '.json', 'w') as f:
    json.dump(point, f)


if __name__ == "__main__":
  py_file1 = sys.argv[1]
  py_file2 = sys.argv[2]
  task_name = sys.argv[3]
  pred_kind = sys.argv[4]

  main()
