import re
import sys
import json
import datapoint
import create_ast

def init(graph):
  # adjust node labels 
  slot_node = None
  for node in graph.get_nodes():
    original_label = node.get('label')
    label = ""
    if 'ast.Str' not in original_label and 'ast.alias' not in original_label and ('name=\'' in original_label or 'id=\'' in original_label):
      result = re.search(r"\'([A-Za-z0-9_@]+)\'", original_label)
      label = result.group(1)
    elif 'object at 0x' in original_label:
      slotted_node = original_label
      slot_node = node
    else:
      label = original_label[4:original_label.find('(')]
    node.set('label', label)

  # assign unique numbers to nodes
  node_to_num = {}
  num = 1
  for node in graph.get_nodes():
    node_to_num[node.get_name()] = num
    num += 1

  slot_node.set('label', graph.get_node(slotted_node)[0].get_label())
  return graph, node_to_num, node_to_num[slot_node.get_name()], node_to_num[slotted_node]

def get_each_edge_category(graph, node_to_num, cat_name):
  edges = []
  for edge in graph.get_edges():
    if edge.get('label') is cat_name:
      n1 = node_to_num[edge.get_source()]
      n2 = node_to_num[edge.get_destination()]
      e = datapoint.GraphEdge(n1, n2)
      edges.append(e)
  return edges

def main(args):
  graph, node_to_num, slot_node_idx, slotted_node_idx = init(create_ast.gen_graph_from_source(args[1], args[2]))

  # prepare edges
  child_edges = get_each_edge_category(graph, node_to_num, 'Child')
  next_token_edges = get_each_edge_category(graph, node_to_num, 'NextToken')
  last_lexical_use_edges = get_each_edge_category(graph, node_to_num, 'LastLexicalUse')
  computed_from_edges = get_each_edge_category(graph, node_to_num, 'ComputedFrom')
  last_use_edges = get_each_edge_category(graph, node_to_num, 'LastRead')
  last_write_edges = get_each_edge_category(graph, node_to_num, 'LastWrite')
  returns_to_edges = get_each_edge_category(graph, node_to_num, 'ReturnsTo')
  edges = datapoint.Edges(
    child=child_edges,
    next_token=next_token_edges,
    last_lexical_use=last_lexical_use_edges,
    computed_from=computed_from_edges,
    last_use=last_use_edges,
    last_write=last_write_edges,
    returns_to=returns_to_edges
  )

  # prepare node_labels
  node_labels_raw = {}
  for node in graph.get_nodes():
    node_labels_raw[node_to_num[node.get_name()]] = node.get('label')

  node_labels = datapoint.NodeLabels(node_labels_raw)

  # prepare context_graph
  context_graph = datapoint.ContextGraph(
    edges=edges,
    node_labels=node_labels
  )

  # create data point
  point = datapoint.DataPoint(
    filename=args[1],
    slot_node_idx=slot_node_idx,
    slotted_node_idx=slotted_node_idx,
    context_graph=context_graph,
    label=args[3]
  )

  with open(args[4], 'w') as outfile:
    json.dump(point.to_dict(), outfile)

if __name__ == "__main__":
  main(sys.argv)
