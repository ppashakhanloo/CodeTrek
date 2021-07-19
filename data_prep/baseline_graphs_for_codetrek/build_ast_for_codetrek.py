import os
import sys
import json
import tempfile
import asttokens

import networkx as nx

from data_prep.baselines.baseline_gnn_ast.diff import get_diff
from data_prep.baselines.baseline_gnn_ast.create_ast import build_child_edges, fix_node_labels

def main():
  graph, neighbors, subtrees, node_of_interest, if_branches, tok1, tok2 = build_child_edges(py_file1, py_file2, task_name, pred_kind)
  graph, terminal_vars, hole_exception = fix_node_labels(graph)

  splits = py_file1.strip().split('/')
  filename = splits[-1]
  prog_label = splits[-2]

  # create map
  node_to_num = dict()
  node_to_pos = dict()
  for idx, node in enumerate(graph.get_nodes()):
    node_to_num[node.get_name()] = (idx + 1, node.get('label') + "(" + str(idx+1) + ")")
    node_to_pos[node.get_name()] = [int(node.get('pos')[1:-1].split(',')[0]), int(node.get('pos')[1:-1].split(',')[1])] if node.get('pos') else None
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
      point = [{
        'anchors': [node_to_num[node_of_interest.get_name()][1]],
        'trajectories': [],
        'hints': [],
        'label': prog_label,
        'source': py_file1
      }]
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
      # get locations of defs
      with tempfile.TemporaryDirectory() as temp_dir:
        #os.system("gsutil cp gs://" + bucket + "/" + py_file1 + " " + temp_dir + "/")
        os.system("gsutil cp gs://" + bucket + "/" + remote_tables_dir + "/" + py_file1 + "/locations_ast.bqrs.csv" + " " + temp_dir + "/")
        os.system("gsutil cp gs://" + bucket + "/" + remote_tables_dir + "/" + py_file1 + "/py_locations.bqrs.csv" + " " + temp_dir + "/")
        os.system("gsutil cp gs://" + bucket + "/" + remote_tables_dir + "/" + py_file1 + "/unused_var.bqrs.csv" + " " + temp_dir + "/")

        locs_ast, py_locs, unused_vars = [], [], []
        with open(temp_dir + "/locations_ast.bqrs.csv", 'r') as f:
          for row in f.readlines()[1:]:
            locs_ast.append(row.strip().split(',')) # loc, mod, ...
        with open(temp_dir + "/py_locations.bqrs.csv", 'r') as f:
          for row in f.readlines()[1:]:
            py_locs.append(row.strip().split(',')) # loc, exp
        with open(temp_dir + "/unused_var.bqrs.csv", 'r') as f:
          for row in f.readlines()[1:]:
            unused_vars.append(row.strip().split(',')) # exp, _, _, var, _

        codeql_results = []
        for uvar in unused_vars:
          for loc_ast in locs_ast:
            for loc in py_locs:
              if uvar[0] == loc[1] and loc[0] == loc_ast[0]:
                codeql_results.append((uvar, [int(loc_ast[2]), int(loc_ast[3])-1]))

        point = []
        for d in defs:
          is_unused = False
          for res in codeql_results:
            if node_to_pos[d[0].get_name()] == res[1]:
              is_unused = True
              break
          if is_unused:
            label = 'unused'
          else:
            label = 'used'
          
          point.append([{
            'anchors': [node_to_num[d[0].get_name()]],
            'trajectories': [],
            'hints': [],
            'label': label,
            'source': py_file1
          }])

  if task_name == 'exception':
    point = [{
      'anchors': [node_to_num[hole_exception.get_name()][1]],
      'trajectories': [],
      'hints': [],
      'label': prog_label,
      'source': py_file1
    }]

  # save stub file
  if pred_kind == 'loc_cls' and task_name == 'defuse':
    index = 0
    for p in point:
      with open('stub_' + filename + '_' + str(index) + '.json', 'w') as f:
        json.dump(p, f)
      index += 1
  else:
    with open('stub_' + filename + '.json', 'w') as f:
       json.dump(point, f)


if __name__ == "__main__":
  bucket = sys.argv[1]
  remote_tables_dir = sys.argv[2]
  py_file1 = sys.argv[3]
  py_file2 = sys.argv[4]
  task_name = sys.argv[5]
  pred_kind = sys.argv[6]

  main()
