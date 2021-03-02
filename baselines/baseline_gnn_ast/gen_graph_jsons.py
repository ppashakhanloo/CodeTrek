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
    value = ""
    if 'object at 0x' in original_label:
      slotted_node = original_label
      slot_node = node
    else:
      label = original_label[4:original_label.find('(')]
      if 'name=' in original_label:
        loc = original_label.find('name=')
        value = original_label[loc+6:original_label.find('\'', loc+6)]
      elif 'id=' in original_label:
        loc = original_label.find('id=')
        value = original_label[loc+4:original_label.find('\'', loc+4)]
      elif 'attr=' in original_label:
        loc = original_label.find('attr=')
        value = original_label[loc+6:original_label.find('\'', loc+6)]
    node.set('label', label+'#'+value)
  
  # assign unique numbers to nodes
  node_to_num = {}
  num = 1
  for node in graph.get_nodes():
    node_to_num[node.get_name()] = num
    num += 1

  slot_node.set('label', 'SlotNode')
  return graph, node_to_num, node_to_num[slot_node.get_name()], node_to_num[slotted_node]

def get_each_edge_category(graph, node_to_num, cat_names):
  edges = {}
  for cat in cat_names:
    edges[cat] = []
  for edge in graph.get_edges():
    if edge.get('label') in cat_names:
      n1 = node_to_num[edge.get_source()]
      n2 = node_to_num[edge.get_destination()]
      e = datapoint.GraphEdge(n1, n2)
      edges[edge.get('label')].append(e)
  return edges

def main(args):
  graph, node_to_num, slot_node_idx, slotted_node_idx = init(create_ast.gen_graph_from_source(args[1], args[2]))
  # prepare edges
  cat_names = ['Child', 'NextToken', 'LastLexicalUse', 'ComputedFrom', 'LastRead', 'LastWrite', 'ReturnsTo']
  all_edges = get_each_edge_category(graph, node_to_num, cat_names)
  edges = datapoint.Edges(
    child=all_edges['Child'],
    next_token=all_edges['NextToken'],
    last_lexical_use=all_edges['LastLexicalUse'],
    computed_from=all_edges['ComputedFrom'],
    last_use=all_edges['LastRead'],
    last_write=all_edges['LastWrite'],
    returns_to=all_edges['ReturnsTo']
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
  
  with open(args[4], 'w', 1000*(2**20)) as outfile:
    json.dump(point.to_dict(), outfile)

if __name__ == "__main__":
  main(sys.argv)
