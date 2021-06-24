import sys
import json
import datapoint
import create_ast
from data_prep.tokenizer import tokenizer

def init(graph, terminal_vars, node_of_interest, hole_exception, terminal_dus, task, pred_kind):
  node_to_num = {}
  for num, node in enumerate(graph.get_nodes()):
    node_to_num[node.get_name()] = num + 1
  if task == 'varmisuse':
    if pred_kind == 'loc_cls':
      slot_node_indexes = [node_of_interest]
    elif pred_kind == 'prog_cls':
      slot_node_indexes = terminal_vars + [node_of_interest]
    else:
      raise NotImplementedError
  elif task == 'defuse':
    if pred_kind == 'loc_cls':
      slot_node_indexes = terminal_dus
    elif pred_kind == 'prog_cls':
      slot_node_indexes = terminal_vars
    else:
      raise NotImplementedError
  elif task == 'exception':
    slot_node_indexes = [hole_exception]
  else:
    raise NotImplementedError

  slot_node_indexes = [node_to_num[n.get_name()] for n in slot_node_indexes]
  return graph, node_to_num, slot_node_indexes

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

def main(args):
  if len(args) != 7:
    print('Usage: python3 gen_graph_jsons.py <file1.py> <file2.py> <label> <output.json> <task> <pred_kind>')
    print('Possible tasks: varmisuse, defuse, exception.')
    print('Possible pred_kinds: prog_cls, loc_cls.')
    exit(1)
  graph, terminal_vars, node_of_interest, hole_exception, terminal_dus = \
          create_ast.gen_graph_from_source(infile=args[1], aux_file=args[2], task_name=args[5])
  graph, node_to_num, slot_node_idxs = \
          init(graph, terminal_vars, node_of_interest, hole_exception, terminal_dus, task=args[5], pred_kind=args[6])
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
    filename=args[1],
    slot_node_idxs=slot_node_idxs,
    context_graph=context_graph,
    label=args[3]
  )
  
  with open(args[4], 'w', 1000*(2**20)) as outfile:
    json.dump(point.to_dict(), outfile)

if __name__ == "__main__":
  main(sys.argv)
