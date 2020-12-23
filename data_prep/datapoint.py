class DataPoint:
    anchor = None      # type: TrajNode
    trajectories = []  # type: List[Trajectory]
    hints = []         # type: List[str]
    label = None       # type: str

    def __init__(self, anchor, trajectories, hints, label):
        self.anchor = anchor
        self.trajectories = trajectories
        self.hints = hints
        self.label = label

    def to_dict(self) -> dict:
        return {
            'anchor': self.anchor.to_dict(),
            'trajectories': [traj.to_dict() for traj in self.trajectories],
            'hints': [hint for hint in self.hints],
            'label': self.label
        }


class Trajectory:
    nodes = []  # type: List[TrajNode]
    edges = []  # type: List[TrajEdge]

    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def to_dict(self) -> dict:
        return {
            'nodes': [node.to_dict() for node in self.nodes],
            'edges': [edge.to_dict() for edge in self.edges]
        }


class TrajNode:
    label = None  # type: str

    def __init__(self, label):
        self.label = label

    def to_dict(self) -> str:
        return self.label


class TrajEdge:
    label1 = None  # type: str
    label2 = None  # type: str

    def __init__(self, label1, label2):
        self.label1 = label1
        self.label2 = label2

    def to_dict(self) -> str:
        return '({l1},{l2})'.format(l1=self.label1, l2=self.label2)
