import sys
from pygraphviz import Node
from random import choice
from data_prep.datapoint import DataPoint, TrajNode
from data_prep.hintutils import HintUtils
from data_prep.walkutils import WalkUtils
from random_walk.randomwalk import RandomWalker
from typing import List


def used_var_hint(walk: List, var_node: Node) -> str:
    return 'pos' if HintUtils.is_used_local_var(walk, var_node) else 'neg'


def main(args: List[str]) -> None:
    if not len(args) == 3:
        print('Usage: python3 random_walk/walk_test.py <gv-file> <out-file>')
        exit(1)

    gv_file = args[1]
    out_file = args[2]
    
    graph = RandomWalker.load_graph_from_gv(gv_file)
    var_nodes = []
    for node in graph.nodes():
        relname, _ = WalkUtils.parse_node_label(graph.nodes[node]['label'])
        if relname == 'variable':
            var_nodes.append(node)
    anchor = choice(var_nodes)
    anchor_label = graph.nodes[anchor]['label']

    walks = RandomWalker.random_walk(graph, anchor, max_num_walks=10, min_num_steps=1, max_num_steps=8)
    while not walks:
        print('Empty walks starting with node:', anchor)
        anchor = choice(var_nodes)
        walks = RandomWalker.random_walk(graph, anchor, max_num_walks=10, min_num_steps=1, max_num_steps=8)

    print('Generated random walks:')
    for walk in walks:
        print(walk)

    # generate the Json file
    source = gv_file
    traj_anchor = TrajNode(anchor_label)
    trajectories = [WalkUtils.build_trajectory(walk) for walk in walks]
    hints = [used_var_hint(walk, anchor_label) for walk in walks]
    # TODO: use the real label
    label = 'used'
    data_point = DataPoint(traj_anchor, trajectories, hints, label, source)
    data_point.dump_json(out_file)
    print('JSON file saved to', out_file)


if __name__ == '__main__':
    main(sys.argv)
