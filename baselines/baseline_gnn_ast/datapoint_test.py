import json
from datapoint import DataPoint, ContextGraph, NodeLabels, Edges, GraphEdge, SymbolCandidate

def main():
  e1 = GraphEdge(0, 1)
  e2 = GraphEdge(3, 5)
  e3 = GraphEdge(3, 8)
  e4 = GraphEdge(1, 8)
  e5 = GraphEdge(5, 8)
  e6 = GraphEdge(9, 8)

  edges = Edges(
    Child=[e1, e2, e3, e4],
    NextToken=[e1, e2, e3],
    LastLexicalUse=[e3, e6, e5],
    LastUse=[e4, e5, e1, e2],
    LastWrite=[e1, e2],
    ReturnsTo=[e6]
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

  cand1 = SymbolCandidate(0, "label0", 'false')
  cand2 = SymbolCandidate(1, "label1", 'true')

  symbol_candidates = [cand1, cand2]

  context_graph = ContextGraph(
    Edges=edges,
    NodeLabels=node_labels
  )

  point = DataPoint(
    filename="test/dir/path",
    slotTokenIdx="0",
    SlotDummyNode="0",
    ContextGraph=context_graph,
    SymbolCandidates=symbol_candidates
  )

  json_str = json.dumps(point.to_dict())
  print(json_str)

if __name__ == '__main__':
    main()
