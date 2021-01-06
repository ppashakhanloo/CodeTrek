from data_prep.walkutils import WalkUtils
from pygraphviz import Node
from typing import List


class HintUtils:

    @staticmethod
    def get_nodes(walk: List) -> List[Node]:
        nodes = []
        for i in range(len(walk)):
            if i % 2 == 0:
                nodes.append(walk[i])
        return nodes

    # over-approximation of whether a local variable (its graph node) is used or not
    # checks if there are at least two occurrences of the variable in the given walk
    @staticmethod
    def is_used_local_var(walk: List, var_node_label: str) -> bool:
        var_relname, var_values = WalkUtils.parse_node_label(var_node_label)
        assert var_relname == 'variable'
        var_id = var_values[0]

        occurrences = set()
        var_id_index = 0
        nodes = HintUtils.get_nodes(walk)
        for node, node_label in nodes:
            relname, values = WalkUtils.parse_node_label(node_label)
            if relname == 'py_variables' and values[var_id_index] == var_id:
                occurrences.add(node)
        return len(occurrences) >= 2
