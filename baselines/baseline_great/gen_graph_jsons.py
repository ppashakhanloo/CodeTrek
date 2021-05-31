import json
import sys

import networkx as nx

from create_ast import gen_graph_from_source


def main(args):
  if len(args) != 6:
    print('Usage: python3 gen_graph_jsons.py <file1.py> <file2.py> <label> <output.json> <task_name>')
    print('Possible task_names: varmisuse, defuse, exception.')
    exit(1)

  file1 = args[1]
  file2 = args[2]
  label = args[3]
  outfile = args[4]
  task_name = args[5]

  corr_except = None
  defuse_label = None
  has_bug = False
  bug_kind = None
  if task_name == "defuse":
    defuse_label = label
    if defuse_label == 'unused': has_bug = True
    bug_kind = 1
  if task_name == "exception":
    corr_except = label
    has_bug = True
    bug_kind = 2
  if task_name == "varmisuse":
    if label == 'misuse': has_bug = True
    bug_kind = 3
  if has_bug:
    bug_kind_name = task_name
  else:
    bug_kind_name = "NONE"

  flat_graph, err_loc, rep_targets, rep_cands, label = gen_graph_from_source(file1, file2, task_name, corr_except=corr_except, defuse_label=defuse_label)

  source_tokens = []
  index = 1
  node_to_num = {}
  for node in sorted(list(flat_graph.nodes(data='loc')), key=lambda x:x[1]):
    source_tokens.append(list({d['tok'] for n, d in flat_graph.nodes.items() if 'tok' in d and d['loc'] == node[1]})[0])
    node_to_num[node[0]] = index
    index += 1

  edges = []
  for edge in flat_graph.edges(data=True):
    src = edge[0]
    dst = edge[1]
    node_type = edge[2]['label']
    node_type_id = edge[2]['id']

    edges.append([node_to_num[src], node_to_num[dst], int(node_type_id), node_type])
  
  point = {
    "has_bug": has_bug,
    "bug_kind": bug_kind,
    "bug_kind_name": bug_kind_name,
    "source_tokens": source_tokens,
    "edges": edges,
    "label": label,
    "error_location": node_to_num[err_loc],
    "repair_targets": [node_to_num[i] for i in rep_targets],
    "repair_candidates": [node_to_num[i] for i in rep_cands],
    "provenances": [{"datasetProvenance": {"datasetName": "cubert", "filepath": file1, "license": "", "note": ""}}]
  }

  with open(outfile, 'w') as f:
    f.write(json.dumps(point))

if __name__ == "__main__":
  main(sys.argv)
