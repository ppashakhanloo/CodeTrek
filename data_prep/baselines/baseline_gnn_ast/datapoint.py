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
  guarded_by = []
  guarded_by_negation = []

  def __init__(self, child, next_token, last_lexical_use,
               computed_from, last_use, last_write, returns_to,
               guarded_by, guarded_by_negation):
    self.child = child
    self.next_token = next_token
    self.last_lexical_use = last_lexical_use
    self.computed_from = computed_from
    self.last_use = last_use
    self.last_write = last_write
    self.returns_to = returns_to
    self.guarded_by = guarded_by
    self.guarded_by_negation = guarded_by_negation

  def to_dict(self):
    return {
        'Child': [edge.to_dict() for edge in self.child],
        'NextToken': [edge.to_dict() for edge in self.next_token],
        'LastLexicalUse': [edge.to_dict() for edge in self.last_lexical_use],
        'ComputedFrom':  [edge.to_dict() for edge in self.computed_from],
        'LastUse': [edge.to_dict() for edge in self.last_use],
        'LastWrite': [edge.to_dict() for edge in self.last_write],
        'ReturnsTo': [edge.to_dict() for edge in self.returns_to],
        'GuardedBy': [edge.to_dict() for edge in self.guarded_by],
        'GuardedByNegation': [edge.to_dict() for edge in self.guarded_by_negation]
    }

class ContextGraph:
  edges = []
  node_types = []
  node_values = []
  node_tokens = []

  def __init__(self, edges, node_types, node_values, node_tokens):
    self.edges = edges
    self.node_types = node_types
    self.node_values = node_values
    self.node_tokens = node_tokens

  def to_dict(self):
    return {
      'Edges': self.edges.to_dict(),
      'NodeTypes': self.node_types,
      'NodeValues': self.node_values,
      'NodeTokens': self.node_tokens
    }

class LocRepDataPoint:
  filename = ""
  slot_node_idxs = []
  context_graph = []
  label = ""

  def __init__(self, filename, errors, repairs, candidates, context_graph, label):
    self.filename = filename
    self.errors = errors
    self.repairs = repairs
    self.candidates = candidates
    self.context_graph = context_graph
    self.label = label

  def to_dict(self):
    return {
      'filename': self.filename,
      'ContextGraph': self.context_graph.to_dict(),
      'label': self.label,
      'errors': self.errors,
      'repairs': self.repairs,
      'candidates': self.candidates
    }

class DataPoint:
  filename = ""
  slot_node_idxs = []
  context_graph = []
  label = ""

  def __init__(self, filename, slot_node_idxs, context_graph, label):
    self.filename = filename
    self.slot_node_idxs = slot_node_idxs
    self.context_graph = context_graph
    self.label = label

  def to_dict(self):
    return {
      'filename': self.filename,
      'SlotNodeIdxs': self.slot_node_idxs,
      'ContextGraph': self.context_graph.to_dict(),
      'label': self.label
    }
  
  def dump_json(self, filepath: str):
    with open(filepath, 'w') as outfile:
      json.dump(self.to_dict(), outfile)

