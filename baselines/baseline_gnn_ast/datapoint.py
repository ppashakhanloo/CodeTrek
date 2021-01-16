import json

class SymbolCandidate:
  SymbolDummyNode = ""
  SymbolName = ""
  IsCorrect = ""

  def __init__(self, SymbolDummyNode, SymbolName, IsCorrect):
    self.SymbolDummyNode = SymbolDummyNode
    self.SymbolName = SymbolName
    self.IsCorrect = IsCorrect
  
  def to_dict(self):
    return {
      'SymbolDummyNode': self.SymbolDummyNode,
      'SymbolName': self.SymbolName,
      'IsCorrect': self.IsCorrect
    }

class GraphEdge:
  src = ""
  dst = ""

  def __init__(self, src, dst):
    self.src = src
    self.dst = dst

  def to_dict(self):
    return [self.src, self.dst]


class Edges:
  NextToken = []
  LastLexicalUse = []
  LastUse = []
  LastWrite = []
  ReturnsTo = []

  def __init__(self, Child, NextToken, LastLexicalUse, LastUse, LastWrite, ReturnsTo):
    self.Child = Child
    self.NextToken = NextToken
    self.LastLexicalUse = LastLexicalUse
    self.LastUse = LastUse
    self.LastWrite = LastWrite
    self.ReturnsTo = ReturnsTo

  def to_dict(self):
    return {
        'Child': [edge.to_dict() for edge in self.Child],
        'NextToken': [edge.to_dict() for edge in self.NextToken],
        'LastLexicalUse': [edge.to_dict() for edge in self.LastLexicalUse],
        'LastUse': [edge.to_dict() for edge in self.LastUse],
        'LastWrite': [edge.to_dict() for edge in self.LastWrite],
        'ReturnsTo': [edge.to_dict() for edge in self.ReturnsTo]
    }

class NodeLabels:
  Labels = {}
  
  def __init__(self, nodeLabels):
    for node_label in nodeLabels:
      self.Labels[node_label] = nodeLabels[node_label]

  def to_dict(self):
    return self.Labels
 
class ContextGraph:
  Edges = []
  NodeLabels = []

  def __init__(self, Edges, NodeLabels):
    self.Edges = Edges
    self.NodeLabels = NodeLabels

  def to_dict(self):
    return {
      'Edges': self.Edges.to_dict(),
      'NodeLabels': self.NodeLabels.to_dict()
    }

class DataPoint:
  filename = ""
  slotTokenIdx = ""
  ContextGraph = []
  SlotDummyNode = ""
  SymbolCandidates = []

  def __init__(self, filename, slotTokenIdx, ContextGraph, SlotDummyNode, SymbolCandidates):
    self.filename = filename
    self.slotTokenIdx = slotTokenIdx
    self.ContextGraph = ContextGraph
    self.SlotDummyNode = SlotDummyNode
    self.SymbolCandidates = SymbolCandidates

  def to_dict(self):
    return {
      'filename': self.filename,
      'slotTokenIdx': self.slotTokenIdx,
      'ContextGraph': self.ContextGraph.to_dict(),
      'SlotDummyNode': self.SlotDummyNode,
      'SymbolCandidates': [cand.to_dict() for cand in self.SymbolCandidates]
    }
  
  def dump_json(self, filepath: str):
    with open(filepath, 'w') as outfile:
      json.dump(self.to_dict(), outfile, indent=4)