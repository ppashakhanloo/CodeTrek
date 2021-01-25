import json
from datapoint import DataPoint, ContextGraph, NodeLabels, Edges, GraphEdge

def main():
  e1 = GraphEdge(0, 1)
  e2 = GraphEdge(3, 5)
  e3 = GraphEdge(3, 8)
  e4 = GraphEdge(1, 8)
  e5 = GraphEdge(5, 8)
  e6 = GraphEdge(9, 8)

  edges = Edges(
    child=[e1, e2, e3, e4],
    next_token=[e1, e2, e3],
    last_lexical_use=[e3, e6, e5],
    last_use=[e4, e5, e1, e2],
    last_write=[e1, e2],
    returns_to=[e6]
  )

  node_labels_raw = {
    0: "label0",
    1: "label1",
    3: "label3",
    5: "label5",
    8: "label8",
    9: "label9"
  }

  node_labels = NodeLabels(node_labels_raw)

  context_graph = ContextGraph(
    edges=edges,
    node_labels=node_labels
  )

  label = 'correct'

  point = DataPoint(
    filename="test/dir/path",
    slot_node="0",
    context_graph=context_graph,
    label=label
  )

  json_str = json.dumps(point.to_dict())
  print(json_str)

if __name__ == '__main__':
    main()
