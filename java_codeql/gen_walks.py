import sys
from random import choice
from typing import List
from data_prep.random_walk.datapoint import DataPoint, AnchorNode
from data_prep.random_walk.walkutils import JavaWalkUtils
from data_prep.random_walk.randomwalk import RandomWalker


def main(args: List[str]) -> None:
    if not len(args) == 3:
        print('Usage: python3 gen_walks.py <gv-file> <out-file>')
        exit(1)

    gv_file = args[1]
    out_file = args[2]

    # Load the graph
    graph = RandomWalker.load_graph_from_gv(gv_file)
    # Randomly pick an anchor from expression nodes
    # TODO: use the real anchor node
    expressions = []
    for node in graph.nodes():
        relname, _ = JavaWalkUtils.parse_node_label(graph.nodes[node]['label'])
        if relname == 'exprs':
            expressions.append(node)
    anchor = choice(expressions)
    anchor_label = graph.nodes[anchor]['label']

    # generate random walks
    language = 'java'
    walker = RandomWalker(graph, anchor_label, language)
    walks = walker.random_walk(max_num_walks=10, min_num_steps=8, max_num_steps=16)

    # generate the Json file
    source = gv_file
    traj_anchor = AnchorNode(anchor_label)
    trajectories = [JavaWalkUtils.build_trajectory(walk) for walk in walks]
    hints = []
    # TODO: use the real label
    label = 'class1'
    data_point = DataPoint(traj_anchor, trajectories, hints, label, source)
    data_point.dump_json(out_file)
    print('JSON file saved to', out_file)


if __name__ == '__main__':
    main(sys.argv)
