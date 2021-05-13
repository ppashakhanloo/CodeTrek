import json
from typing import List, Dict


class AnchorNode:
    label = None  # type: str

    def __init__(self, label: str):
        self.label = label

    def to_dict(self) -> str:
        return self.label


class TrajNodeType:
    node_type = None  # type: str

    def __init__(self, node_type: str):
        self.node_type = node_type

    def to_dict(self) -> str:
        return self.node_type

class TrajNodeValue:
    node_value = None # type: str

    def __init__(self, node_value: str):
        self.node_value = node_value

    def to_dict(self) -> str:
        return self.node_value

class TrajEdge:
    label1 = None  # type: str
    label2 = None  # type: str

    def __init__(self, label1: str, label2: str):
        self.label1 = label1
        self.label2 = label2

    def to_dict(self) -> str:
        return '({l1},{l2})'.format(l1=self.label1, l2=self.label2)


class Trajectory:
    node_types = []  # type: List[TrajNodeType]
    node_values = []  # type: List[TrajNodeValue]
    edges = []  # type: List[TrajEdge]

    def __init__(self, node_types: List[TrajNodeType], node_values: List[TrajNodeValue], edges: List[TrajEdge]):
        self.node_types = node_types
        self.node_values = node_values
        self.edges = edges

    def to_dict(self) -> Dict:
        return {
            'node_types': [node.to_dict() for node in self.node_types],
            'node_values': [node.to_dict() for node in self.node_values],
            'edges': [edge.to_dict() for edge in self.edges]
        }


class DataPoint:
    anchor = None      # type: AnchorNode
    trajectories = []  # type: List[Trajectory]
    hints = []         # type: List[str]
    label = None       # type: str
    source = None      # type: str

    def __init__(self, anchor: AnchorNode, trajectories: List[Trajectory], hints: List[str], label: str, source: str):
        self.anchor = anchor
        self.trajectories = trajectories
        self.hints = hints
        self.label = label
        self.source = source

    def to_dict(self) -> Dict:
        return {
            'anchor': self.anchor.to_dict(),
            'trajectories': [traj.to_dict() for traj in self.trajectories],
            'hints': [hint for hint in self.hints],
            'label': self.label,
            'source': self.source
        }

    def dump_json(self, filepath: str) -> None:
        with open(filepath, 'w') as outfile:
            json.dump(self.to_dict(), outfile)
