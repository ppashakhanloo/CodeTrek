import sys
import networkx as nx
from pygraphviz import AGraph
from random import choice
from random_walk import random_walk
from data_prep.datapoint import DataPoint, TrajNode
from data_prep.walkutils import WalkUtils
from typing import List


def load_graph_from_gv(path: str) -> AGraph:
    graph = AGraph(path, directed=False)
    return nx.nx_agraph.from_agraph(graph)


def main(args: List[str]) -> None:
    if not len(args) == 3:
        print('Usage: python3 random_walk/walk_test.py <gv-file> <out-file>')
        exit(1)

    gv_file = args[1]
    out_file = args[2]
    
    graph = load_graph_from_gv(gv_file)
    # TODO: use the real anchor node
    node = choice(list(graph.nodes()))
    walks = random_walk(graph, node, max_num_walks=10, min_num_steps=1,
        max_num_steps=8)
    while not walks:
        print('Empty walks starting with node:', node)
        node = choice(list(graph.nodes()))
        walks = random_walk(graph, node, max_num_walks=10, min_num_steps=1,
            max_num_steps=8)

    print('Generated random walks:')
    for walk in walks:
        print(walk)

    # generate the Json file
    source = gv_file
    anchor = TrajNode(graph.nodes[node]['label'])
    trajectories = [WalkUtils.parse_trajectory(walk) for walk in walks]
    # TODO: use the real hints
    hints = []
    # TODO: use the real label
    label = 'used'
    data_point = DataPoint(anchor, trajectories, hints, label, source)
    data_point.dump_json(out_file)
    print('JSON file saved to', out_file)


if __name__ == '__main__':
    main(sys.argv)
