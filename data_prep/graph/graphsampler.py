import queue
from graphviz import Graph
from pygraphviz import AGraph, Node
from typing import Set


class GraphSampler:

    @staticmethod
    def sample(agraph: AGraph, anchor_id: str, node_num: int) -> Graph:
        nodes = agraph.nodes()
        graph = Graph()
        q = queue.SimpleQueue()  # type: SimpleQueue
        q.put(anchor_id)
        visited = set()          # type: Set[str]
        # sample the nodes in breadth-first order
        while not q.empty() and len(visited) < node_num:
            name = q.get()       # type: str
            if name not in visited:
                visited.add(name)
                graph.node(name, nodes[name]['label'])
                for neighbor in agraph.neighbors(name):
                    if neighbor not in visited:
                        q.put(neighbor)
        # find all edges between sampled nodes
        for (n1, n2) in agraph.edges():
            if n1 in visited and n2 in visited:
                labels = {edge['label'] for edge in list(agraph[n1][n2].values())}
                for label in labels:
                    graph.edge(n1, n2, label)
        return graph
