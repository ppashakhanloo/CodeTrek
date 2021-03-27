import sys
from data_prep.graph.graphbuilder import GraphBuilder
from typing import List


def main(args: List[str]) -> None:
    if not len(args) == 4:
        print('Usage: python3 build_graph.py <facts_dir> <join_filepath> <output_file>')
        exit(1)

    facts_dir = args[1]
    join_filepath = args[2]
    output_file = args[3]

    language = 'python'
    graph_builder = GraphBuilder(facts_dir, join_filepath, language)
    # When include_callgraph=True then callgraphs are added to the graph.
    # The default is include_callgraph=False.
    graph = graph_builder.build(include_callgraph=True)
    GraphBuilder.save_gv(graph, output_file + '.gv')


if __name__ == "__main__":
    main(sys.argv)
