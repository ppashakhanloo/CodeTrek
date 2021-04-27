import sys
import json
from data_prep.graph.graphbuilder import GraphBuilder
from data_prep.graph.graphsampler import GraphSampler
from data_prep.graph.graphutils import GraphUtils
from typing import List


def get_anchor_from_stub_file(filepath: str) -> str:
    with open(filepath) as f:
        stub = json.load(f)
    return stub[0]['anchor']


def main(args: List[str]) -> None:
    if not len(args) == 5:
        print('Usage: python3 sample_graph.py <graph_file> <stub_file> <node_num> <out_file>')
        exit(1)

    graph_file = args[1]     # type: str
    stub_file = args[2]      # type: str
    node_num = int(args[3])  # type: int
    out_file = args[4]       # type: str

    # Load graph from the .gv file
    agraph = GraphUtils.load_graph_from_gv(graph_file)
    # Find the anchor node by stub label
    anchor_label = get_anchor_from_stub_file(stub_file)
    anchor = GraphUtils.find_node_by_label(agraph, anchor_label)
    # Sample the sub-graph
    graph = GraphSampler.sample(agraph, anchor, node_num)
    # Save the sub-graph to output file
    GraphBuilder.save_gv(graph, out_file)


if __name__ == "__main__":
    main(sys.argv)
