from data_prep.datapoint import TrajNode, TrajEdge, Trajectory
from pygraphviz import Node
from typing import List, Dict, Tuple


class WalkUtils:

    expr_kinds = {  # type: Dict[int, str]
        0: 'py_Attribute',
        1: 'py_BinaryExpr',
        2: 'py_BoolExpr',
        3: 'py_Bytes',
        4: 'py_Call',
        5: 'py_ClassExpr',
        6: 'py_Compare',
        7: 'py_Dict',
        8: 'py_DictComp',
        9: 'py_Ellipsis',
        10: 'py_FunctionExpr',
        11: 'py_GeneratorExp',
        12: 'py_IfExp',
        13: 'py_ImportExpr',
        14: 'py_ImportMember',
        15: 'py_Lambda',
        16: 'py_List',
        17: 'py_ListComp',
        18: 'py_Name',
        19: 'py_Num',
        20: 'py_Repr',
        21: 'py_Set',
        22: 'py_SetComp',
        23: 'py_Slice',
        24: 'py_Starred',
        25: 'py_Str',
        26: 'py_Subscript',
        27: 'py_Tuple',
        28: 'py_UnaryExpr',
        29: 'py_Yield',
        30: 'py_YieldFrom',
        31: 'py_TemplateDottedNotation',
        32: 'py_Filter',
        33: 'py_PlaceHolder',
        34: 'py_Await',
        35: 'py_Fstring',
        36: 'py_FormattedValue',
        37: 'py_AssignExpr',
        38: 'py_SpecialOperation'
    }

    stmt_kinds = {  # type: Dict[int, str]
        0: 'py_Assert',
        1: 'py_Assign',
        2: 'py_AugAssign',
        3: 'py_Break',
        4: 'py_Continue',
        5: 'py_Delete',
        6: 'py_ExceptStmt',
        7: 'py_Exec',
        8: 'py_Expr_stmt',
        9: 'py_For',
        10: 'py_Global',
        11: 'py_If',
        12: 'py_Import',
        13: 'py_ImportStar',
        14: 'py_Nonlocal',
        15: 'py_Pass',
        16: 'py_Print',
        17: 'py_Raise',
        18: 'py_Return',
        19: 'py_Try',
        20: 'py_While',
        21: 'py_With',
        22: 'py_TemplateWrite',
        23: 'py_AnnAssign'
    }

    @staticmethod
    def gen_node_label(relname: str, values: List[str]) -> str:
        # Use variable ID as the label
        if relname == 'variable':
            return 'v_' + values[0]
        # Distinguish different kinds of expressions
        if relname == 'py_exprs':
            kind_index = 1
            kind = int(values[kind_index])
            return 'expr_' + WalkUtils.expr_kinds[kind]
        # Distinguish different kinds of statements
        if relname == 'py_stmts':
            kind_index = 1
            kind = int(values[kind_index])
            return 'stmt_' + WalkUtils.stmt_kinds[kind]
        # Otherwise, use relation name as the label
        return relname

    @staticmethod
    def parse_node(node: Tuple[Node, str]) -> TrajNode:
        orig_label = node[1]  # text: relname(val1,val2,...)
        splits = orig_label.split('(')
        assert len(splits) == 2
        relname = splits[0].strip()
        tokens = splits[1].strip()[:-1].split(',')
        values = [token.strip() for token in tokens]
        return TrajNode(WalkUtils.gen_node_label(relname, values))

    @staticmethod
    def parse_edge(edge: Tuple[Node, Node, str]) -> TrajEdge:
        label = edge[2]  # text: (label1,label2)
        tokens = label.split(',')
        assert len(tokens) == 2
        label1 = tokens[0].strip()[1:]
        label2 = tokens[1].strip()[:-1]
        return TrajEdge(label1, label2)

    @staticmethod
    def parse_trajectory(walk: List[Tuple]) -> Trajectory:
        nodes = []
        edges = []
        for i in range(len(walk)):
            if i % 2 == 0:  # node
                nodes.append(WalkUtils.parse_node(walk[i]))
            else:           # edge
                edges.append(WalkUtils.parse_edge(walk[i]))
        return Trajectory(nodes, edges)
