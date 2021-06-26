import os
import sys
import json
import tempfile
import datapoint
import create_ast
import multiprocessing
from joblib import Parallel, delayed
from data_prep.tokenizer import tokenizer

def get_each_edge_category(graph, node_to_num, cat_names):
  edges = {}
  for cat in cat_names:
    edges[cat] = []
  graph_edges = graph.get_edges()
  for edge in graph_edges:
    if edge.get('label') in cat_names:
      n1 = node_to_num[edge.get_source()]
      n2 = node_to_num[edge.get_destination()]
      e = datapoint.GraphEdge(n1, n2)
      edges[edge.get('label')].append(e)
  return edges

def create_sample(graph, anchor_indexes, label, source, node_to_num):
  # prepare edges
  cat_names = ['Child', 'NextToken', 'LastLexicalUse', 'ComputedFrom',
               'LastRead', 'LastWrite', 'ReturnsTo', 'GuardedBy', 'GuardedByNegation']
  all_edges = get_each_edge_category(graph, node_to_num, cat_names)
  edges = datapoint.Edges(
    child=all_edges['Child'],
    next_token=all_edges['NextToken'],
    last_lexical_use=all_edges['LastLexicalUse'],
    computed_from=all_edges['ComputedFrom'],
    last_use=all_edges['LastRead'],
    last_write=all_edges['LastWrite'],
    returns_to=all_edges['ReturnsTo'],
    guarded_by=all_edges['GuardedBy'],
    guarded_by_negation=all_edges['GuardedByNegation']
  )

  # prepare node_types, node_values, and node tokens
  node_types = [0] * len(node_to_num.keys())
  node_values = [0] * len(node_types)
  node_tokens = [0] * len(node_types)
  for node in node_to_num.keys():
    splits = graph.get_node(node)[0].get('label').split('[SEP]')
    if len(splits) == 2: # the node has type and value
      node_types[node_to_num[node]-1] = splits[0]
      node_values[node_to_num[node]-1] = splits[1]
      node_tokens[node_to_num[node]-1] = tokenizer.tokenize(splits[1], 'python')
    else: # the node only has type
      node_types[node_to_num[node]-1] = splits[0]
      node_values[node_to_num[node]-1] = ""
      node_tokens[node_to_num[node]-1] = splits[0]

  # prepare context_graph
  context_graph = datapoint.ContextGraph(
    edges=edges,
    node_types=node_types,
    node_values=node_values,
    node_tokens=node_tokens
  )

  # create data point
  point = datapoint.DataPoint(
    filename=source,
    slot_node_idxs=anchor_indexes,
    context_graph=context_graph,
    label=label
  )

  return point.to_dict()

def gen_varmisuse(pred_kind):
    pass

def gen_exception(path):
  remote_tables_dir = "gs://" + bucket_name + "/" + remote_table_dirname + "/" + path
  splits = path.split('/')
  filename = splits[-1]
  prog_label = splits[-2]

  try:
    with tempfile.TemporaryDirectory() as tables_dir:
      os.system("gsutil cp " + "gs://" + bucket_name + "/" + path + " " + tables_dir)
      g, terminal_vars, node_of_interest, hole_exception = \
            create_ast.gen_graph_from_source(infile=tables_dir + "/" + filename, aux_file=None, \
            task_name='exception')
      node_to_num = {}
      for num, node in enumerate(g.get_nodes()):
        node_to_num[node.get_name()] = num + 1
      anchor_node = [node_to_num[hole_exception.get_name()]]
      with open(tables_dir + "/graph_" + filename + ".json", 'w') as f:
        json.dump(create_sample(g, anchor_node, prog_label, path, node_to_num), f)
      os.system("gsutil cp " + tables_dir + "/graph_" + filename + ".json" + " " + \
                "gs://" + bucket_name + "/" + output_graphs_dirname + "/" + \
                path.replace('/' + prog_label, '').replace(filename, ''))
    with open(tables_paths_file + "-done", "a") as done:
      done.write(path + "\n")
  except Exception as e:
    with open(tables_paths_file + "-log", "a") as log:
      log.write(">>" + path + str(e) + "\n")

def gen_defuse(path, pred_kind):
  remote_tables_dir = "gs://" + bucket_name + "/" + remote_table_dirname + "/" + path
  splits = path.split('/')
  filename = splits[-1]
  prog_label = splits[-2]

  try:
    with tempfile.TemporaryDirectory() as tables_dir:
      os.system("gsutil cp " + remote_tables_dir + "/unused_var.bqrs.csv" + " " + tables_dir)
      os.system("gsutil cp " + "gs://" + bucket_name + "/" + path + " " + tables_dir)

      unused_var_names = []
      with open(tables_dir + "/unused_var.bqrs.csv", 'r') as f:
        for line in f.readlines():
          v = line.strip().split(',')[3][1:-1]
          unused_var_names.append(v)

      g, terminal_vars, node_of_interest, hole_exception = \
              create_ast.gen_graph_from_source(infile=tables_dir + "/" + filename, aux_file=None, \
                                               task_name='defuse', pred_kind=pred_kind)
      node_to_num = {}
      for num, node in enumerate(g.get_nodes()):
        node_to_num[node.get_name()] = num + 1
      anchor_nodes = []
      if pred_kind == 'prog_cls':
        for v in terminal_vars:
          if v[2] == 'write':
            anchor_nodes.append(v[0])
        anchor_nodes = [node_to_num[n.get_name()] for n in anchor_nodes]
        with open(tables_dir + "/graph_" + filename + ".json", 'w') as f:
          json.dump(create_sample(g, anchor_nodes, prog_label, path, node_to_num), f)
        os.system("gsutil cp " + tables_dir + "/graph_" + filename + ".json" + " " + \
                  "gs://" + bucket_name + "/" + output_graphs_dirname + "/" + path.replace('/' + prog_label, ''))
      elif pred_kind == 'loc_cls':
        write_terminal_vars_unused = {}
        write_terminal_vars_used = {}
        for v in terminal_vars:
          if v[1] in unused_var_names and v[2] == 'write':
            if v[1] in write_terminal_vars_unused:
              write_terminal_vars_unused[v[1]].append(v)
            else:
              write_terminal_vars_unused[v[1]] = [v]
          elif v[2] == 'write':
            if v[1] in write_terminal_vars_used:
              write_terminal_vars_used[v[1]].append(v)
            else:
              write_terminal_vars_used[v[1]] = [v]
        for var_name in write_terminal_vars_unused:
          for idx, n in enumerate(write_terminal_vars_unused[var_name]):
            with open(tables_dir + "/graph_" + var_name + '_' + str(idx) + '_' + filename + ".json", 'w') as f:
              json.dump(create_sample(g, [node_to_num[n[0].get_name()]], 'unused', path, node_to_num), f)
            os.system("gsutil cp " + tables_dir + "/graph_" + var_name + '_' + str(idx) + '_' + filename + ".json" + " " + \
                      "gs://" + bucket_name + "/" + output_graphs_dirname + "/" + path.replace(prog_label+'/'+filename, ''))
        for var_name in write_terminal_vars_used:
          for idx, n in enumerate(write_terminal_vars_used[var_name]):
            with open(tables_dir + "/graph_" + var_name + '_' + str(idx) + '_' + filename + ".json", 'w') as f:
              json.dump(create_sample(g, [node_to_num[n[0].get_name()]], 'used', path, node_to_num), f)
            os.system("gsutil cp " + tables_dir + "/graph_" + var_name + '_' + str(idx) + '_' + filename + ".json" + " " + \
                      "gs://" + bucket_name + "/" + output_graphs_dirname + "/" + path.replace(prog_label+'/'+filename, ''))
      else:
        raise NotImplementedError(pred_kind)
    with open(tables_paths_file + "-done", "a") as done:
      done.write(path + "\n")
  except Exception as e:
    with open(tables_paths_file + "-log", "a") as log:
      log.write(">>" + path + str(e) + "\n")

if __name__ == "__main__":
  tables_paths_file = sys.argv[1] # paths.txt
  bucket_name = sys.argv[2] # generated-tables
  remote_table_dirname = sys.argv[3] # outdir_reshuffle
  output_graphs_dirname = sys.argv[4] # output_graphs
  task = sys.argv[5] # defuse, exception, varmisuse
  pred_kind = sys.argv[6]

  paths = []
  with open(tables_paths_file, 'r') as fin:
    for line in fin.readlines():
      paths.append(line.strip())

  if task == 'varmisuse':
    gen_varmisuse(pred_kind)
  elif task == 'defuse':
    Parallel(n_jobs=10, prefer="threads")(delayed(gen_defuse)(path, pred_kind) for path in paths)
  elif task == 'exception':
    Parallel(n_jobs=10, prefer="threads")(delayed(gen_exception)(path) for path in paths)
  else:
    raise NotImplementedError(task)
