import json
from typing import List, Dict


class TrajNode:
    label = None  # type: str

    def __init__(self, label: str):
        self.label = label

    def to_dict(self) -> str:
        return self.label


class TrajEdge:
    label1 = None  # type: str
    label2 = None  # type: str

    def __init__(self, label1: str, label2: str):
        self.label1 = label1
        self.label2 = label2

    def to_dict(self) -> str:
        return '({l1},{l2})'.format(l1=self.label1, l2=self.label2)


class Trajectory:
    nodes = []  # type: List[TrajNode]
    edges = []  # type: List[TrajEdge]

    def __init__(self, nodes: List[TrajNode], edges: List[TrajEdge]):
        self.nodes = nodes
        self.edges = edges

    def to_dict(self) -> Dict:
        return {
            'nodes': [node.to_dict() for node in self.nodes],
            'edges': [edge.to_dict() for edge in self.edges]
        }


class DataPoint:
    anchor = None      # type: TrajNode
    trajectories = []  # type: List[Trajectory]
    hints = []         # type: List[str]
    label = None       # type: str
    source = None      # type: str

    def __init__(self, anchor: TrajNode, trajectories: List[Trajectory], hints: List[str], label: str, source: str):
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
            json.dump(self.to_dict(), outfile, indent=4)
