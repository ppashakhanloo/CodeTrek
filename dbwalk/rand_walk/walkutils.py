from pygraphviz import Node
from typing import List, Dict, Tuple
from dbwalk.rand_walk.datapoint import TrajNode, TrajEdge, Trajectory


class WalkUtils:

    COLUMNS = {
        'variable': ['id', 'scope', 'name'],
        'locations_ast': ['id', 'module', 'beginLine', 'beginColumn', 'endLine', 'endColumn'],
        'py_Classes': ['id', 'parent'],
        'py_Functions': ['id', 'parent'],
        'py_Modules': ['id'],
        'py_boolops': ['id', 'kind', 'parent'],
        'py_bytes': ['id', 'parent', 'idx'],
        'py_cmpops': ['id', 'kind', 'parent', 'idx'],
        'py_cmpop_lists': ['id', 'parent'],
        'py_comprehensions': ['id', 'parent', 'idx'],
        'py_comprehension_lists': ['id', 'parent'],
        'py_dict_items': ['id', 'kind', 'parent', 'idx'],
        'py_dict_item_lists': ['id', 'parent'],
        'py_exprs': ['id', 'kind', 'parent', 'idx'],
        'py_expr_contexts': ['id', 'kind', 'parent'],
        'py_expr_lists': ['id', 'parent', 'idx'],
        'py_ints': ['id', 'parent'],
        'py_locations': ['id', 'parent'],
        'py_numbers': ['id', 'parent', 'idx'],
        'py_operators': ['id', 'kind', 'parent'],
        'py_parameter_lists': ['id', 'parent'],
        'py_stmts': ['id', 'kind', 'parent', 'idx'],
        'py_stmt_lists': ['id', 'parent', 'idx'],
        'py_strs': ['id', 'parent', 'idx'],
        'py_str_lists': ['id', 'parent'],
        'py_unaryops': ['id', 'kind', 'parent'],
        'py_variables': ['id', 'parent'],
        'py_successors': ['predecessor', 'successor'],
        'py_true_successors': ['predecessor', 'successor'],
        'py_exception_successors': ['predecessor', 'successor'],
        'py_false_successors': ['predecessor', 'successor'],
        'py_flow_bb_node': ['flownode', 'realnode', 'basicblock', 'index'],
        'py_scope_flow': ['flow', 'scope', 'kind'],
        'py_idoms': ['node', 'immediate_dominator'],
        'py_scopes': ['node', 'scope'],
        'py_scope_location': ['id', 'scope'],
        'py_ssa_phi': ['phi', 'arg'],
        'py_ssa_var': ['id', 'var'],
        'py_ssa_use': ['node', 'var'],
        'py_ssa_defn': ['id', 'node']
    }

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
    def parse_node_label(node_label: str) -> Tuple[str, List[str]]:
        # text of node_label: relname(val1,val2,...)
        # only split on the first occurrence of '('
        # since it is possible that one of the elements will contain the same character
        splits = node_label.split('(', 1)
        assert len(splits) == 2
        relname = splits[0].strip()
        if relname == 'py_strs':        # special case where there may be unquoted commas in the string
            tokens = splits[1].strip()[:-1].rsplit(',', 2)
        else:
            tokens = splits[1].strip()[:-1].split(',')
        assert len(tokens) == len(WalkUtils.COLUMNS[relname])
        values = [token.strip() for token in tokens]
        return relname, values

    @staticmethod
    def build_traj_node(node: Tuple[Node, str]) -> TrajNode:
        node_label = node[1]
        relname, values = WalkUtils.parse_node_label(node_label)
        return TrajNode(WalkUtils.gen_node_label(relname, values))

    @staticmethod
    def parse_edge_label(edge_label: str) -> Tuple[str, str]:
        # text of edge_label: (label1,label2)
        tokens = edge_label.split(',')
        assert len(tokens) == 2
        label1 = tokens[0].strip()[1:]
        label2 = tokens[1].strip()[:-1]
        return label1, label2

    @staticmethod
    def build_traj_edge(edge: Tuple[Node, Node, str]) -> TrajEdge:
        edge_label = edge[2]
        label1, label2 = WalkUtils.parse_edge_label(edge_label)
        return TrajEdge(label1, label2)

    @staticmethod
    def build_trajectory(walk: List[Tuple]) -> Trajectory:
        nodes = []
        edges = []
        for i in range(len(walk)):
            if i % 2 == 0:  # node
                nodes.append(WalkUtils.build_traj_node(walk[i]))
            else:           # edge
                edges.append(WalkUtils.build_traj_edge(walk[i]))
        return Trajectory(nodes, edges)


class JavaWalkUtils:

    COLUMNS = {
        'locations_default': ['id', 'file', 'beginLine', 'beginColumn', 'endLine', 'endColumn'],
        'hasLocation': ['locatableid', 'id'],
        'packages': ['id', 'nodeName'],
        'primitives': ['id', 'nodeName'],
        'modifiers': ['id', 'nodeName'],
        'classes': ['id', 'nodeName', 'parentid', 'sourceid'],
        'idRecord': ['id'],
        'interfaces': ['id', 'nodeName', 'parentid', 'sourceid'],
        'fielddecls': ['id', 'parentid'],
        'fieldDeclaredIn': ['fieldId', 'fieldDeclId', 'pos'],
        'fields': ['id', 'nodeName', 'typeid', 'parentid', 'sourceid'],
        'constrs': ['id', 'nodeName', 'signature', 'typeid', 'parentid', 'sourceid'],
        'methods': ['id', 'nodeName', 'signature', 'typeid', 'parentid', 'sourceid'],
        'params': ['id', 'typeid', 'pos', 'parentid', 'sourceid'],
        'paramName': ['id', 'nodeName'],
        'isVarargsParam': ['param'],
        'exceptions': ['id', 'typeid', 'parentid'],
        'isAnnotType': ['interfaceid'],
        'isAnnotElem': ['methodid'],
        'annotValue': ['parentid', 'id2', 'value'],
        'isEnumType': ['classid'],
        'isEnumConst': ['fieldid'],
        'typeVars': ['id', 'nodeName', 'pos', 'kind', 'parentid'],
        'wildcards': ['id', 'nodeName', 'kind'],
        'typeBounds': ['id', 'typeid', 'pos', 'parentid'],
        'typeArgs': ['argumentid', 'pos', 'parentid'],
        'isParameterized': ['memberid'],
        'isRaw': ['memberid'],
        'erasure': ['memberid', 'erasureid'],
        'isAnonymClass': ['classid', 'parent'],
        'isLocalClass': ['classid', 'parent'],
        'isDefConstr': ['constructorid'],
        'lambdaKind': ['exprId', 'bodyKind'],
        'arrays': ['id', 'nodeName', 'elementtypeid', 'dimension', 'componenttypeid'],
        'enclInReftype': ['child', 'parent'],
        'extendsReftype': ['id1', 'id2'],
        'implInterface': ['id1', 'id2'],
        'hasModifier': ['id1', 'id2'],
        'imports': ['id1', 'holder', 'name', 'kind'],
        'stmts': ['id', 'kind', 'parent', 'idx', 'bodydecl'],
        'exprs': ['id', 'kind', 'typeid', 'parent', 'idx'],
        'callableEnclosingExpr': ['id', 'callable_id'],
        'statementEnclosingExpr': ['id', 'statement_id'],
        'isParenthesized': ['id', 'parentheses'],
        'callableBinding': ['callerid', 'callee'],
        'memberRefBinding': ['id', 'callable'],
        'variableBinding': ['expr', 'variable'],
        'localvars': ['id', 'nodeName', 'typeid', 'parentid'],
        'namestrings': ['name', 'value', 'parent']
    }

    expr_kinds = {  # type: Dict[int, str]
        1: 'arrayaccess',
        2: 'arraycreationexpr',
        3: 'arrayinit',
        4: 'assignexpr',
        5: 'assignaddexpr',
        6: 'assignsubexpr',
        7: 'assignmulexpr',
        8: 'assigndivexpr',
        9: 'assignremexpr',
        10: 'assignandexpr',
        11: 'assignorexpr',
        12: 'assignxorexpr',
        13: 'assignlshiftexpr',
        14: 'assignrshiftexpr',
        15: 'assignurshiftexpr',
        16: 'booleanliteral',
        17: 'integerliteral',
        18: 'longliteral',
        19: 'floatingpointliteral',
        20: 'doubleliteral',
        21: 'characterliteral',
        22: 'stringliteral',
        23: 'nullliteral',
        24: 'mulexpr',
        25: 'divexpr',
        26: 'remexpr',
        27: 'addexpr',
        28: 'subexpr',
        29: 'lshiftexpr',
        30: 'rshiftexpr',
        31: 'urshiftexpr',
        32: 'andbitexpr',
        33: 'orbitexpr',
        34: 'xorbitexpr',
        35: 'andlogicalexpr',
        36: 'orlogicalexpr',
        37: 'ltexpr',
        38: 'gtexpr',
        39: 'leexpr',
        40: 'geexpr',
        41: 'eqexpr',
        42: 'neexpr',
        43: 'postincexpr',
        44: 'postdecexpr',
        45: 'preincexpr',
        46: 'predecexpr',
        47: 'minusexpr',
        48: 'plusexpr',
        49: 'bitnotexpr',
        50: 'lognotexpr',
        51: 'castexpr',
        52: 'newexpr',
        53: 'conditionalexpr',
        54: 'parexpr',
        55: 'instanceofexpr',
        56: 'localvariabledeclexpr',
        57: 'typeliteral',
        58: 'thisaccess',
        59: 'superaccess',
        60: 'varaccess',
        61: 'methodaccess',
        62: 'unannotatedtypeaccess',
        63: 'arraytypeaccess',
        64: 'packageaccess',
        65: 'wildcardtypeaccess',
        66: 'declannotation',
        67: 'uniontypeaccess',
        68: 'lambdaexpr',
        69: 'memberref',
        70: 'annotatedtypeaccess',
        71: 'typeannotation',
        72: 'intersectiontypeaccess',
        73: 'switchexpr'
    }

    stmt_kinds = {  # type: Dict[int, str]
        0: 'block',
        1: 'ifstmt',
        2: 'forstmt',
        3: 'enhancedforstmt',
        4: 'whilestmt',
        5: 'dostmt',
        6: 'trystmt',
        7: 'switchstmt',
        8: 'synchronizedstmt',
        9: 'returnstmt',
        10: 'throwstmt',
        11: 'breakstmt',
        12: 'continuestmt',
        13: 'emptystmt',
        14: 'exprstmt',
        15: 'labeledstmt',
        16: 'assertstmt',
        17: 'localvariabledeclstmt',
        18: 'localclassdeclstmt',
        19: 'constructorinvocationstmt',
        20: 'superconstructorinvocationstmt',
        21: 'case',
        22: 'catchclause',
        23: 'yieldstmt'
    }

    @staticmethod
    def gen_node_label(relname: str, values: List[str]) -> str:
        # Use variable ID as the label
        if relname == 'localvars':
            return 'v_' + values[0]
        # Distinguish different kinds of expressions
        if relname == 'exprs':
            kind_index = 1
            kind = int(values[kind_index])
            return 'expr_' + JavaWalkUtils.expr_kinds[kind]
        # Distinguish different kinds of statements
        if relname == 'stmts':
            kind_index = 1
            kind = int(values[kind_index])
            return 'stmt_' + JavaWalkUtils.stmt_kinds[kind]
        # Otherwise, use relation name as the label
        return relname

    @staticmethod
    def parse_node_label(node_label: str) -> Tuple[str, List[str]]:
        # text of node_label: relname(val1,val2,...)
        # only split on the first occurrence of '('
        splits = node_label.split('(', 1)
        assert len(splits) == 2
        relname = splits[0].strip()
        tokens = splits[1].strip()[:-1].split(',')
        assert len(tokens) == len(JavaWalkUtils.COLUMNS[relname])
        values = [token.strip() for token in tokens]
        return relname, values

    @staticmethod
    def build_traj_node(node: Tuple[Node, str]) -> TrajNode:
        node_label = node[1]
        relname, values = JavaWalkUtils.parse_node_label(node_label)
        return TrajNode(JavaWalkUtils.gen_node_label(relname, values))

    @staticmethod
    def parse_edge_label(edge_label: str) -> Tuple[str, str]:
        # text of edge_label: (label1,label2)
        tokens = edge_label.split(',')
        assert len(tokens) == 2
        label1 = tokens[0].strip()[1:]
        label2 = tokens[1].strip()[:-1]
        return label1, label2

    @staticmethod
    def build_traj_edge(edge: Tuple[Node, Node, str]) -> TrajEdge:
        edge_label = edge[2]
        label1, label2 = JavaWalkUtils.parse_edge_label(edge_label)
        return TrajEdge(label1, label2)

    @staticmethod
    def build_trajectory(walk: List[Tuple]) -> Trajectory:
        nodes = []
        edges = []
        for i in range(len(walk)):
            if i % 2 == 0:  # node
                nodes.append(JavaWalkUtils.build_traj_node(walk[i]))
            else:           # edge
                edges.append(JavaWalkUtils.build_traj_edge(walk[i]))
        return Trajectory(nodes, edges)
