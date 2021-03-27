import sys
from typing import List
from data_prep.graph.graphutils import GraphUtils


def main(args: List[str]) -> None:
    if not len(args) == 4:
        print('Usage: python3 shortest_path.py <graph-file> <from-label> <to-label>')
        exit(1)

    gv_file = args[1]
    from_label = args[2]
    to_label = args[3]

    graph = GraphUtils.load_graph_from_gv(gv_file)
    path = GraphUtils.shortest_path(graph, from_label, to_label)
    for node_id in path:
        print(f'node: {node_id}, label: {GraphUtils.find_label_by_id(graph, node_id)}')


if __name__ == '__main__':
    main(sys.argv)
