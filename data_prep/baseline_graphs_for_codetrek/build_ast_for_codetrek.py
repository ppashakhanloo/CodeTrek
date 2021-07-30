import os
import sys
import json
import random
import tempfile
import asttokens

import networkx as nx
from joblib import Parallel, delayed

from data_prep.utils.utils import read_csv, read_lines
from data_prep.utils.gcp_utils import gcp_copy_from, gcp_copy_to
from data_prep.baselines.baseline_gnn_ast.diff import get_diff
from data_prep.baselines.baseline_gnn_ast.create_ast import build_child_edges, fix_node_labels

def gen_graph(path):
  splits = path.split('/')
  filename = splits[-1]
  prog_label = splits[-2]
  num = filename[5:-3]

  py_file1 = path
  py_file2 = None
  if prog_label == 'correct':
     py_file2 = path.replace(num, str(int(num)+1)).replace('/correct/', '/misuse/')
  if prog_label == 'misuse':
     py_file2 = path.replace(num, str(int(num)-1)).replace('/misuse/', '/correct/')

  try:
    temp_dir = tempfile.TemporaryDirectory()
    gcp_copy_from(output_graphs_dir + '/' + path.replace(prog_label + '/' + filename, '') + 'stub_' + filename + '.json', temp_dir.name, bucket)
    if os.path.exists(temp_dir.name + '/stub_' + filename + '.json'):
      raise Exception('already exists.')

    gcp_copy_from(py_file1, temp_dir.name, bucket)
    py_file1 = os.path.join(temp_dir.name, filename)
    if py_file2:
      gcp_copy_from(py_file2, temp_dir.name, bucket)
      py_file2 = os.path.join(temp_dir.name, py_file2.split('/')[-1])

    graph, neighbors, subtrees, node_of_interest, if_branches, tok1, tok2, node_to_pos = build_child_edges(py_file1, py_file2, task_name, pred_kind)
    graph, terminal_vars, hole_exception = fix_node_labels(graph)

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
    with open(temp_dir.name + '/graph_' + filename + '.gv', 'w') as f:
      f.write(new_graph)
    gcp_copy_to(temp_dir.name + '/graph_' + filename + '.gv', output_graphs_dir + '/' + path.replace(prog_label + '/' + filename, ''), bucket)

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
          'source': path
        }]
      if pred_kind == 'loc_cls':
        point = [{
          'anchors': [node_to_num[node_of_interest.get_name()][1]],
          'trajectories': [],
          'hints': [],
          'label': prog_label,
          'source': path
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
          'source': path
        }]
      if pred_kind == 'loc_cls':
        # get locations of defs
        gcp_copy_from(remote_tables_dir + "/" + path + "/locations_ast.bqrs.csv", temp_dir.name, bucket)
        gcp_copy_from(remote_tables_dir + "/" + path + "/py_locations.bqrs.csv", temp_dir.name, bucket)
        gcp_copy_from(remote_tables_dir + "/" + path + "/unused_var.bqrs.csv", temp_dir.name, bucket)

        locs_ast = read_csv(temp_dir.name + '/locations_ast.bqrs.csv')
        py_locs = read_csv(temp_dir.name + '/py_locations.bqrs.csv')
        unused_vars = read_csv(temp_dir.name + '/unused_var.bqrs.csv')

        codeql_results = []
        for uvar in unused_vars:
          for loc_ast in locs_ast:
            for loc in py_locs:
              if uvar[0] == loc[1] and loc[0] == loc_ast[0]:
                codeql_results.append((uvar, [int(loc_ast[2]), int(loc_ast[3])-1]))

        point = []
        for d in defs:
          label = 'used'
          for res in codeql_results:
            if d[0].get_name().endswith('_"'):
              search_name = d[0].get_name()[1:-2]
            else:
              search_name = d[0].get_name()
            if node_to_pos[search_name] == res[1]:
              label = 'unused'
              break

          point.append([{
            'anchors': [node_to_num[d[0].get_name()][1]],
            'trajectories': [],
            'hints': [],
            'label': label,
            'source': path
          }])

    if task_name == 'exception':
      point = [{
        'anchors': [node_to_num[hole_exception.get_name()][1]],
        'trajectories': [],
        'hints': [],
        'label': prog_label,
        'source': path
      }]

    if task_name == 'shadow_global':
      point = [{
        'anchors': [node_to_num[n[0].get_name()][1] for n in random.choices(terminal_vars, k=min(len(terminal_vars), 10))],
        'trajectories': [],
        'hints': [],
        'label': prog_label,
        'source': path
      }]

    # save stub file
    with open(temp_dir.name + '/stub_' + filename + '.json', 'w') as f:
      json.dump(point, f)
    gcp_copy_to(temp_dir.name + '/stub_' + filename + '.json', output_graphs_dir + '/' + path.replace(prog_label + '/' + filename, ''), bucket)

    with open(paths_file + '_done', 'a') as f:
      f.write(path + '\n')
  except Exception as e:
    with open(paths_file + '_error', 'a') as f:
      f.write(path + ' ' + str(e) + '\n')


if __name__ == "__main__":
  bucket = sys.argv[1]
  remote_tables_dir = sys.argv[2]
  output_graphs_dir = sys.argv[3]
  paths_file = sys.argv[4]
  task_name = sys.argv[5]
  pred_kind = sys.argv[6]

  Parallel(n_jobs=10, prefer="threads")(delayed(gen_graph)(path) for path in read_lines(paths_file))
