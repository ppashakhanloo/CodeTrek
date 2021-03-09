import json

class GraphEdge:
  src = ""
  dst = ""

  def __init__(self, src, dst):
    self.src = src
    self.dst = dst

  def to_dict(self):
    return [self.src, self.dst]


class Edges:
  child = []
  next_token = []
  last_lexical_use = []
  computed_from = []
  last_use = []
  last_write = []
  returns_to = []

  def __init__(self, child, next_token, last_lexical_use, computed_from, last_use, last_write, returns_to):
    self.child = child
    self.next_token = next_token
    self.last_lexical_use = last_lexical_use
    self.computed_from = computed_from
    self.last_use = last_use
    self.last_write = last_write
    self.returns_to = returns_to

  def to_dict(self):
    return {
        'Child': [edge.to_dict() for edge in self.child],
        'NextToken': [edge.to_dict() for edge in self.next_token],
        'LastLexicalUse': [edge.to_dict() for edge in self.last_lexical_use],
        'ComputedFrom':  [edge.to_dict() for edge in self.computed_from],
        'LastUse': [edge.to_dict() for edge in self.last_use],
        'LastWrite': [edge.to_dict() for edge in self.last_write],
        'ReturnsTo': [edge.to_dict() for edge in self.returns_to]
    }

class NodeLabels:
  labels = {}
  
  def __init__(self, node_labels):
    for node_label in node_labels:
      self.labels[node_label] = node_labels[node_label]

  def to_dict(self):
    return self.labels
 
class ContextGraph:
  edges = []
  node_labels = []

  def __init__(self, edges, node_labels):
    self.edges = edges
    self.node_labels = node_labels

  def to_dict(self):
    return {
      'Edges': self.edges.to_dict(),
      'NodeLabels': self.node_labels.to_dict()
    }

class DataPoint:
  filename = ""
  slot_node_idx = ""
  context_graph = []
  label = ""

  def __init__(self, filename, slot_node_idx, context_graph, label):
    self.filename = filename
    self.slot_node_idx = slot_node_idx
    self.context_graph = context_graph
    self.label = label

  def to_dict(self):
    return {
      'filename': self.filename,
      'SlotNodeIdx': self.slot_node_idx,
      'ContextGraph': self.context_graph.to_dict(),
      'label': self.label
    }
  
  def dump_json(self, filepath: str):
    with open(filepath, 'w') as outfile:
      json.dump(self.to_dict(), outfile, indent=4)
