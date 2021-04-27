import networkx as nx
import pickle
from graphviz import Graph
from pygraphviz import AGraph, Node
from typing import List


class GraphUtils:

    @staticmethod
    def load_graph_from_gv(path: str) -> AGraph:
        graph = AGraph(path, directed=False)
        return nx.nx_agraph.from_agraph(graph)

    @staticmethod
    def deserialize(in_path: str) -> AGraph:
        infile = open(in_path, mode='rb')
        agraph = pickle.load(infile)
        infile.close()
        return agraph

    @staticmethod
    def save_gv(graph: Graph, output_file: str) -> None:
        with open(output_file, 'w') as outfile:
            outfile.write(str(graph))

    @staticmethod
    def serialize(graph: Graph, out_path: str):
        agraph = AGraph(graph.source, directed=False)
        xgraph = nx.nx_agraph.from_agraph(agraph)
        outfile = open(out_path, mode='wb')
        pickle.dump(xgraph, outfile)
        outfile.close()

    @staticmethod
    def find_node_by_label(graph: AGraph, node_label: str) -> str:
        for node in graph.nodes():
            if graph.nodes()[node]['label'] == node_label:
                return node
        raise NameError('Cannot find node with label:' + node_label)

    @staticmethod
    def find_label_by_id(graph: AGraph, id: str) -> str:
        return graph.nodes()[id]['label']

    @staticmethod
    def shortest_path(graph: AGraph, from_label: str, to_label: str) -> List[str]:
        from_node = GraphUtils.find_node_by_label(graph, from_label)
        to_node = GraphUtils.find_node_by_label(graph, to_label)
        return nx.shortest_path(graph, source=from_node, target=to_node)
