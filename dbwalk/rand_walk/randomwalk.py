import networkx as nx
from pygraphviz import AGraph, Node
from random import choice, choices, randint
from typing import List, Dict, Set
from dbwalk.rand_walk.walkutils import WalkUtils


class RandomWalker:
    BIAS_WEIGHT = 5
    PYTHON_BIAS_TABLES = {'py_variables', 'py_exprs', 'py_stmts'}
    JAVA_BIAS_TABLES = {'exprs', 'stmts'}

    language = None         # type: str
    graph = None            # type: AGraph
    source_node = None      # type: Node
    node_to_relname = None  # type: Dict[Node, str]
    bias_tables = None      # type: Set[str]

    def __init__(self, graph: AGraph, source: str, language: str):
        self.language = RandomWalker.parse_language(language)
        self.graph = graph
        self.source_node = RandomWalker.find_node_by_label(graph, source)
        self.node_to_relname = RandomWalker.build_relname_map(graph)
        self.bias_tables = RandomWalker.load_bias_tables(self.language)

    @staticmethod
    def find_node_by_label(graph: AGraph, node_label: str) -> Node:
        for node in graph.nodes():
            if graph.nodes()[node]['label'] == node_label:
                return node
        raise NameError('Cannot find node with label:' + node_label)

    @staticmethod
    def parse_language(language: str) -> str:
        if language.lower() == 'python':
            return 'python'
        elif language.lower() == 'java':
            return 'java'
        else:
            raise ValueError('Invalid language:', language)

    @staticmethod
    def load_bias_tables(language: str) -> Set[str]:
        if language == 'python':
            return RandomWalker.PYTHON_BIAS_TABLES
        elif language == 'java':
            return RandomWalker.JAVA_BIAS_TABLES
        else:
            raise ValueError('Unknown bias tables for', language)

    # Sample the next node to visit for a random walk.
    # The probability of visiting a node in `bias_tables` is
    # BIAS_WEIGHT times as that of visiting other nodes.
    def sample_node(self, nodes: List[Node], node_to_relname: Dict[Node, str]) -> Node:
        weights = []
        for node in nodes:
            relname = node_to_relname[node]
            weight = RandomWalker.BIAS_WEIGHT if relname in self.bias_tables else 1
            weights.append(weight)
        return choices(nodes, weights, k=1)[0]

    # The output is a list of simple walks starting from the specified node.
    # Each walk: [(n1, n1_lbl), (n1, n2, e1_lbl), (n2, n2_lbl), ...]
    #             _____^______  _______^_______   _____^_____
    #                node            edge             node
    # The length of walks can be different and is specified by
    # min_num_steps and max_num_steps.
    # The number of random walks can be less than max_num_walks if there are less
    # walks from the source than the requested number of walks.
    # If that happens, this function pads the return list to max_num_walks by
    # duplicating elements randomly chosen from the sampled walks.
    def random_walk(self, max_num_walks: int, min_num_steps: int, max_num_steps: int) -> List[List]:
        walks = list()
        walk = 0
        while walk < max_num_walks * 3 and len(walks) < max_num_walks:
            walk += 1
            curr_walk = [(self.source_node, self.graph.nodes[self.source_node]['label'])]
            curr_node = self.source_node
            curr_edge = None
            random_num_steps = randint(min_num_steps, max_num_steps)

            for step in range(random_num_steps):
                neighbors = list(self.graph.neighbors(curr_node))
                if len(neighbors) > 0:
                    prev_node = curr_node
                    curr_node = self.sample_node(neighbors, self.node_to_relname)
                    curr_edge = (prev_node, curr_node,
                                 choice(list(self.graph[prev_node][curr_node].values()))['label'])
                    curr_edge_rev = (curr_edge[1], curr_edge[0], curr_edge[2])

                    if curr_edge not in curr_walk \
                            and curr_edge_rev not in curr_walk \
                            and (curr_node, self.graph.nodes[curr_node]['label']) not in curr_walk:
                        curr_walk.append(curr_edge)
                        curr_walk.append((curr_node, self.graph.nodes[curr_node]['label']))
                    else:
                        curr_node = prev_node

            if curr_walk not in walks and len(curr_walk) > 1:
                walks.append(curr_walk)

        return RandomWalker.padding(walks, max_num_walks)

    @staticmethod
    def build_relname_map(graph: AGraph) -> Dict[Node, str]:
        node_to_relname = {}
        for node in graph.nodes():
            relname, _ = WalkUtils.parse_node_label(graph.nodes[node]['label'])
            node_to_relname[node] = relname
        return node_to_relname

    @staticmethod
    def load_graph_from_gv(path: str) -> AGraph:
        graph = AGraph(path, directed=False)
        return nx.nx_agraph.from_agraph(graph)

    # Padding the given list to size `num` by duplicating elements that are
    # selected from the list at random.
    @staticmethod
    def padding(lst: List[any], num: int) -> List[any]:
        assert len(lst) <= num
        if len(lst) == num:
            return lst
        else:  # len(lst) < num
            count = num - len(lst)
            return lst + choices(lst, k=count)
