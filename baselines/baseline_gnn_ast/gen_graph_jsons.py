import re
import sys
import json
import datapoint
import create_ast

def init(graph):
  # adjust node labels  
  for node in graph.get_nodes():
    original_label = node.get('label')
    label = ""
    if 'name=\'' in original_label or 'id=\'' in original_label:
      result = re.search(r"\'([A-Za-z0-9_]+)\'", original_label)
      label = result.group(1)
    else:
      label = original_label[4:original_label.find('(')]
    node.set('label', label)

  # assign unique numbers to nodes
  node_to_num = {}
  num = 1
  for node in graph.get_nodes():
    node_to_num[node.get_name()] = num
    num += 1

  return graph, node_to_num

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
  graph, node_to_num = init(create_ast.gen_graph_from_source(args[1]))

  # prepare edges
  Child_edges = get_each_edge_category(graph, node_to_num, 'Child')
  NextToken_edges = get_each_edge_category(graph, node_to_num, 'NextToken')
  LastLexicalUse_edges = get_each_edge_category(graph, node_to_num, 'LastLexicalUse')
  LastUse_edges = get_each_edge_category(graph, node_to_num, 'LastRead')
  LastWrite_edges = get_each_edge_category(graph, node_to_num, 'LastWrite')
  ReturnsTo_edges = get_each_edge_category(graph, node_to_num, 'ReturnsTo')
  edges = datapoint.Edges(
    Child=Child_edges,
    NextToken=NextToken_edges,
    LastLexicalUse=LastLexicalUse_edges,
    LastUse=LastUse_edges,
    LastWrite=LastWrite_edges,
    ReturnsTo=ReturnsTo_edges
  )

  # prepare node_labels
  node_labels_raw = {}
  for node in graph.get_nodes():
    node_labels_raw[node_to_num[node.get_name()]] = node.get('label')

  node_labels = datapoint.NodeLabels(node_labels_raw)

  # prepare candidates
  symbol_candidates = []
  cand_correct = SymbolCandidate(1, "correct_variable", 'true')
  cand_incorrect = SymbolCandidate(2, "incorrect_variable", 'false')
  symbol_candidates = [cand_correct, cand_incorrect]

  # prepare context_graph
  context_graph = datapoint.ContextGraph(
    Edges=edges,
    NodeLabels=node_labels
  )

  # create data point
  point = datapoint.DataPoint(
    filename=args[1],
    slotTokenIdx="TODO",
    SlotDummyNode="0",
    ContextGraph=context_graph,
    SymbolCandidates=symbol_candidates
  )

  js = json.dumps(point.to_dict(), indent=4)
  print(js)

if __name__ == "__main__":
  main(sys.argv)
