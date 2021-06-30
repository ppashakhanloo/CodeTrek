import os
import sys
import json
import tempfile

import networkx as nx
from create_ast import gen_graph_from_source
from joblib import Parallel, delayed

bug_kinds = {'bug-free': 0, 'varmisuse': 1, 'defuse': 2, 'exception': 3}

def compute_graph(file1, file2, task_name, pred_kind):
  flat_graph, err_loc, rep_targets, rep_cands, defs = gen_graph_from_source(file1, file2, task_name, pred_kind)
  assert flat_graph, "flat_graph should not be null."
  source_tokens = []
  index = 1
  node_to_num = {}
  for node in sorted(list(flat_graph.nodes(data=True)), key=lambda x:x[1]['loc'] if 'loc' in x[1] else (0,0)):
    if len(node) > 0 and 'tok' in node[1]:
      if node[1]['tok']:
        source_tokens.append(node[1]['tok'])
      else:
        source_tokens.append('')
    else:
      source_tokens.append('')
    node_to_num[node[0]] = index
    index += 1

  edges = []
  for edge in flat_graph.edges(data=True):
    src = edge[0]
    dst = edge[1]
    node_type = edge[2]['label']
    node_type_id = edge[2]['id']
    edges.append([node_to_num[src], node_to_num[dst], int(node_type_id), node_type])

  return source_tokens, edges, node_to_num, err_loc, rep_targets, rep_cands, defs

def gen_exception(path):
  filename = path[path.rfind('/')+1:]
  prog_label = path.split('/')[-2]
  with tempfile.TemporaryDirectory() as tables_dir:
    os.system("gsutil cp gs://" + bucket_name + "/" + path + " " + tables_dir)
    source_tokens, edges, node_to_num, _, _, _, _ = compute_graph(tables_dir + "/" + filename, None, 'exception', None)
    point = {
      "has_bug": True,
      "bug_kind": bug_kinds['exception'],
      "bug_kind_name": 'exception',
      "source_tokens": source_tokens,
      "edges": edges,
      "label": prog_label,
      "provenances": [{"datasetProvenance": {"datasetName": "cubert", "filepath": path, "license": "null", "note": "null"}}]
    }
    with open(tables_dir + '/graph_' + filename + '.json', 'w') as f:
      json.dump(point, f)
    os.system('gsutil cp ' + tables_dir + '/graph_' + filename + '.json' + ' '\
              'gs://' + bucket_name + '/' + output_graphs_dirname + '/' + path.replace(prog_label + '/' + filename, ''))

def gen_defuse(path, pred_kind):
  filename = path[path.rfind('/')+1:]
  prog_label = path.split('/')[-2]
  with tempfile.TemporaryDirectory() as tables_dir:
    os.system("gsutil cp gs://" + bucket_name + "/" + path + " " + tables_dir)
    source_tokens, edges, node_to_num, _, _, _, defs = compute_graph(tables_dir + "/" + filename, None, 'defuse', pred_kind)
    if pred_kind == 'prog_cls':
      point = {
        "has_bug": True if prog_label == 'unused' else False,
        "bug_kind": bug_kinds['defuse'],
        "bug_kind_name": 'defuse',
        "source_tokens": source_tokens,
        "edges": edges,
        "label": prog_label,
        "provenances": [{"datasetProvenance": {"datasetName": "cubert", "filepath": path, "license": "null", "note": "null"}}]
      }
      with open(tables_dir + '/graph_' + filename + '.json', 'w') as f:
        json.dump(point, f)
      os.system('gsutil cp ' + tables_dir + '/graph_' + filename + '.json' + ' ' + \
                'gs://' + bucket_name + '/' + output_graphs_dirname + '/' + path.replace(prog_label + '/' + filename, ''))
    if pred_kind == 'loc_cls':
      os.system("gsutil cp gs://" + bucket_name + "/" + remote_table_dirname + "/" + path + "/unused_var.bqrs.csv " + tables_dir)
      unused_defs = set()
      with open(tables_dir + '/unused_var.bqrs.csv', 'r') as f:
        for row in f.readlines()[1:]:
            unused_defs.add(row.strip().split(',')[3][1:-1])
      for def_v in defs:
        if def_v[-1] in unused_defs:
          loc_label = 'unused'
        else:
          loc_label = 'used'
        point = {
          "has_bug": True if loc_label == 'unused' else False,
          "bug_kind": bug_kinds[task_name],
          "bug_kind_name": task_name,
          "source_tokens": source_tokens,
          "edges": edges,
          "label": loc_label,
          "error_location": node_to_num[def_v[0]] if loc_label == 'unused' else len(node_to_num) + 1,
          "provenances": [{"datasetProvenance": {"datasetName": "cubert", "filepath": path, "license": "null", "note": "null"}}]
        }
        with open(tables_dir + '/graph_' + def_v[-1] + '_' + filename + '.json', 'w') as f:
          json.dump(point, f)
        os.system("gsutil cp " + tables_dir + "/graph_" + def_v[-1] + '_' + filename + ".json" + " " + \
                  "gs://" + bucket_name + "/" + output_graphs_dirname + "/" + path.replace(prog_label + '/' + filename, ''))

def gen_varmisuse(path, pred_kind):
  filename = path[path.rfind('/')+1:]
  prog_label = path.split('/')[-2]

  if pred_kind == 'prog_cls':
    os.system("gsutil cp gs://" + bucket_name + "/" + path + " " + tempfile.gettempdir() + "/")
    source_tokens, edges, node_to_num, err_loc, rep_targets, rep_cands, _ = compute_graph(tempfile.gettempdir() + "/" + filename, None, 'varmisuse', pred_kind)
    point = {
      "has_bug": True if prog_label == 'misuse' else False,
      "bug_kind": bug_kinds['varmisuse'],
      "bug_kind_name": task_name,
      "source_tokens": source_tokens,
      "edges": edges,
      "label": prog_label,
      "provenances": [{"datasetProvenance": {"datasetName": "cubert", "filepath": path, "license": "null", "note": "null"}}]
    }
    with open(tempfile.gettempdir() + '/graph_' + filename + '.json', 'w') as f:
      json.dump(point, f)
    os.system('gsutil cp ' + tempfile.gettempdir() + '/graph_' + filename + '.json' + ' ' + \
              'gs://' + bucket_name + '/' + output_graphs_dirname + '/' + path.replace(prog_label + '/' + filename, ''))
    return

  with tempfile.TemporaryDirectory() as tables_dir:
    num = int(py_file[5:-3])
    if prog_label == "correct":
      file_1 = "gs://" + bucket_name + "/varmisuse/" + category + "/" + "correct" + "/" + "file_" + str(num) + ".py"
      file_1_src = "file_" + str(num) + ".py"
      file_2 = "gs://" + bucket_name + "/varmisuse/" + category + "/" + "misuse" + "/" + "file_" + str(num + 1) + ".py"
      file_2_src = "file_" + str(num + 1) + ".py"
    else:
      file_1 = "gs://" + bucket_name + "/varmisuse/" + category + "/" + "misuse" + "/" + "file_" + str(num) + ".py"
      file_1_src = "file_" + str(num) + ".py"
      file_2 = "gs://" + bucket_name + "/varmisuse/" + category + "/" + "correct" + "/" + "file_" + str(num - 1) + ".py"
      file_2_src = "file_" + str(num - 1) + ".py"
    os.system("gsutil cp  " + file_1 + " " + tables_dir + "/" + file_1_src)
    os.system("gsutil cp  " + file_2 + " " + tables_dir + "/" + file_2_src)
    remote_tables_dir = "gs://" + bucket_name + "/" + remote_table_dirname + "/" + tables_path
    diff_bin = os.path.join(home_path, 'data_prep/random_walk/diff.py')
    os.system("gsutil -m cp  -r " + remote_tables_dir + "/*" + " " + tables_dir)
    os.system("python " + diff_bin + " " + tables_dir + "/" + file_1_src + " " + tables_dir + "/" + file_2_src + " " + tables_dir + "/var_misuses.csv")


    source_tokens, edges, node_to_num, err_loc, rep_targets, rep_cands, _ = compute_graph(tables_dir + "/" + filename, None, 'defuse', pred_kind)

# wip
def main():
  if task_name == "varmisuse":
    if pred_kind == 'loc_rep':
      point = {
        "has_bug": True if label == 'misuse' else False,
        "bug_kind": bug_kinds[task_name],
        "bug_kind_name": task_name,
        "source_tokens": source_tokens,
        "edges": edges,
        "label": label,
        "error_location": node_to_num[err_loc] if label == 'misuse' else len(node_to_num) + 1,
        "repair_targets": [node_to_num[i] for i in rep_targets] if label == 'misuse' else [],
        "repair_candidates": [node_to_num[i] for i in rep_cands],
        "provenances": [{"datasetProvenance": {"datasetName": "cubert", "filepath": file1, "license": "null", "note": "null"}}]
      }
    elif pred_kind == 'prog_cls':
      point = {
        "has_bug": True if label == 'misuse' else False,
        "bug_kind": bug_kinds[task_name],
        "bug_kind_name": task_name,
        "source_tokens": source_tokens,
        "edges": edges,
        "label": label,
        "provenances": [{"datasetProvenance": {"datasetName": "cubert", "filepath": file1, "license": "null", "note": "null"}}]
      }
    elif pred_kind == 'loc_cls':
      err_location = None
      loc_label = "NULL"
      point = {
        "has_bug": True if loc_label == 'misuse' else False,
        "bug_kind": bug_kinds[task_name],
        "bug_kind_name": task_name,
        "source_tokens": source_tokens,
        "edges": edges,
        "location": err_location,
        "label": loc_label,
        "provenances": [{"datasetProvenance": {"datasetName": "cubert", "filepath": file1, "license": "null", "note": "null"}}]
      }


if __name__ == "__main__":
  tables_paths_file = sys.argv[1] # paths.txt
  bucket_name = sys.argv[2] # generated-tables
  remote_table_dirname = sys.argv[3] # outdir_reshuffle
  output_graphs_dirname = sys.argv[4] # output_graphs
  task_name = sys.argv[5] # defuse, exception, varmisuse
  assert task_name in ['defuse', 'exception', 'varmisuse']
  pred_kind = sys.argv[6]
  assert pred_kind in ['prog_cls', 'loc_cls', 'loc_rep']

  paths = []
  with open(tables_paths_file, 'r') as fin:
    for line in fin.readlines():
      paths.append(line.strip())

  if task_name == 'varmisuse':
    Parallel(n_jobs=10, prefer="threads")(delayed(gen_varmisuse)(path, pred_kind) for path in paths)
  if task_name == 'defuse':
    Parallel(n_jobs=10, prefer="threads")(delayed(gen_defuse)(path, pred_kind) for path in paths)
  if task_name == 'exception':
    Parallel(n_jobs=10, prefer="threads")(delayed(gen_exception)(path) for path in paths)
