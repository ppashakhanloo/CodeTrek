import ast

import asttokens
import networkx as nx

from diff import get_diff

CURR_STR = 'HOLE'

def op_to_symbol(op):
  if isinstance(op, ast.Add):
    return '+'
  if isinstance(op, ast.Sub):
    return '-'
  if isinstance(op, ast.Mult):
    return '*'
  if isinstance(op, ast.MatMult):
    return '@'
  if isinstance(op, ast.Div):
    return '/'
  if isinstance(op, ast.Mod):
    return '%'
  if isinstance(op, ast.Pow):
    return '**'
  if isinstance(op, ast.LShift):
    return '<<'
  if isinstance(op, ast.RShift):
    return '>>'
  if isinstance(op, ast.BitOr):
    return '|'
  if isinstance(op, ast.BitXor):
    return '^'
  if isinstance(op, ast.BitAnd):
    return '&'
  if isinstance(op, ast.FloorDiv):
    return '//'
  if isinstance(op, ast.Not):
    return 'not'
  if isinstance(op, ast.Invert):
    return '~'
  if isinstance(op, ast.UAdd):
    return '+='
  if isinstance(op, ast.USub):
    return '-='
  if isinstance(op, ast.And):
    return 'and'
  if isinstance(op, ast.Or):
    return 'or'

def get_AST_nodes(contents):
  nodes_AST = {}
  assigns = {}
  subtrees = {}
  ifs_info = {}

  atok = asttokens.ASTTokens(contents, parse=True)
  for node in ast.walk(atok.tree):
    if not hasattr(node, 'lineno'):
      continue
    elif isinstance(node, ast.FunctionDef):
      nodes_AST[(node.lineno, node.col_offset)] = (node, 'def ' + node.name)
    elif isinstance(node, ast.Attribute):
      nodes_AST[(node.lineno, node.col_offset)] = (node, '.')
    elif isinstance(node, ast.Tuple):
      nodes_AST[(node.lineno, node.col_offset)] = (node, '(')
    elif isinstance(node, ast.Subscript):
      nodes_AST[(node.lineno, node.col_offset)] = (node, '[')
    elif hasattr(node, 'lineno') and hasattr(node, 'id'):
      nodes_AST[(node.lineno, node.col_offset)] = (node, node.id)
    elif hasattr(node, 'lineno') and hasattr(node, 'name'):
      nodes_AST[(node.lineno, node.col_offset)] = (node, node.name)
    elif hasattr(node, 'lineno') and hasattr(node, 'arg'):
      nodes_AST[(node.lineno, node.col_offset)] = (node, node.arg)
    elif isinstance(node, ast.Return):
      nodes_AST[(node.lineno, node.col_offset)] = (node, 'return')
    elif isinstance(node, ast.Delete):
      nodes_AST[(node.lineno, node.col_offset)] = (node, 'delete')
    elif isinstance(node, ast.For):
      nodes_AST[(node.lineno, node.col_offset)] = (node, 'for')
    elif isinstance(node, ast.If):
      nodes_AST[(node.lineno, node.col_offset)] = (node, 'if')
    elif isinstance(node, ast.While):
      nodes_AST[(node.lineno, node.col_offset)] = (node, 'while')
    elif isinstance(node, ast.Try):
      nodes_AST[(node.lineno, node.col_offset)] = (node, 'try')
    elif isinstance(node, ast.ExceptHandler):
      nodes_AST[(node.lineno, node.col_offset)] = (node, 'except')
    elif isinstance(node, ast.Raise):
      nodes_AST[(node.lineno, node.col_offset)] = (node, 'raise')
    elif isinstance(node, ast.Expr):
      nodes_AST[(node.lineno, node.col_offset)] = (node, 'expr')
    elif isinstance(node, ast.Assign):
      nodes_AST[(node.lineno, node.col_offset)] = (node, 'assign')
    elif isinstance(node, ast.Assert):
      nodes_AST[(node.lineno, node.col_offset)] = (node, 'assert')
    elif isinstance(node, ast.Str):
      nodes_AST[(node.lineno, node.col_offset)] = (node, "\'"+node.s.replace('\n','\\n').replace('\t', '\\t')+"\'")
    elif isinstance(node, ast.Call):
      name = None
      if isinstance(node.func, ast.Attribute):
        if isinstance(node.func.value, str):
          name = node.func.value
      elif isinstance(node.func, ast.Subscript):
        if isinstance(node.func.value, str):
          name = node.func.value
      elif isinstance(node.func, ast.Name):
        if isinstance(node.func.id, str):
          name = node.func.id
      else:
        name = "call"
      nodes_AST[(node.lineno, node.col_offset)] = (node, name)
    elif isinstance(node, ast.NameConstant):
      nodes_AST[(node.lineno, node.col_offset)] = (node, str(node.value))
    elif isinstance(node, ast.Num):
      nodes_AST[(node.lineno, node.col_offset)] = (node, str(node.n))
    elif isinstance(node, ast.UnaryOp):
      nodes_AST[(node.lineno, node.col_offset)] = (node, op_to_symbol(node.op))
    elif isinstance(node, ast.BoolOp):
      nodes_AST[(node.lineno, node.col_offset)] = (node, op_to_symbol(node.op))
    elif isinstance(node, ast.BinOp):
      nodes_AST[(node.lineno, node.col_offset)] = (node, op_to_symbol(node.op))
    elif isinstance(node, ast.List) or isinstance(node, ast.ListComp):
      nodes_AST[(node.lineno, node.col_offset)] = (node, 'list')
    elif isinstance(node, ast.Dict):
      nodes_AST[(node.lineno, node.col_offset)] = (node, 'dict')
    elif isinstance(node, ast.Set):
      nodes_AST[(node.lineno, node.col_offset)] = (node, 'set')
    elif isinstance(node, ast.IfExp):
      nodes_AST[(node.lineno, node.col_offset)] = (node, 'if')
    elif isinstance(node, ast.Pass):
      nodes_AST[(node.lineno, node.col_offset)] = (node, 'pass')
    elif isinstance(node, ast.Yield):
      nodes_AST[(node.lineno, node.col_offset)] = (node, 'yield')
    elif isinstance(node, ast.Continue):
      nodes_AST[(node.lineno, node.col_offset)] = (node, 'continue')
    elif isinstance(node, ast.Break):
      nodes_AST[(node.lineno, node.col_offset)] = (node, 'break')
    elif isinstance(node, ast.With):
      nodes_AST[(node.lineno, node.col_offset)] = (node, 'with')
    elif isinstance(node, ast.Lambda):
      nodes_AST[(node.lineno, node.col_offset)] = (node, 'lambda')
    elif isinstance(node, ast.Raise):
      nodes_AST[(node.lineno, node.col_offset)] = (node, 'raise')
    elif isinstance(node, ast.Try):
      nodes_AST[(node.lineno, node.col_offset)] = (node, 'try')
    elif isinstance(node, ast.Assert):
      nodes_AST[(node.lineno, node.col_offset)] = (node, 'assert')
    elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
      nodes_AST[(node.lineno, node.col_offset)] = (node, 'import')
    elif isinstance(node, ast.Global):
      nodes_AST[(node.lineno, node.col_offset)] = (node, 'global')

    if isinstance(node, ast.AugAssign):
      assigns[(node.lineno, node.col_offset)] = (node, '=')
      subtrees[node.value] = []
    if isinstance(node, ast.Assign):
      assigns[(node.lineno, node.col_offset)] = (node, '=')
      subtrees[node.value] = []

    if isinstance(node, ast.If):
      ifs_info[node] = {'test': [], 'body': []}

  for val in subtrees:
    for node in ast.walk(val):
      if isinstance(node, ast.Name):
        subtrees[val].append(node)
  
  for if_node in ifs_info:
    for node in ast.walk(if_node.test):
      if isinstance(node, ast.Name):
        ifs_info[if_node]['test'].append(node)

    for bb in if_node.body + if_node.orelse:
      for node in ast.walk(bb):
        if isinstance(node, ast.Name): ifs_info[if_node]['body'].append(node)

  return nodes_AST, assigns, subtrees, ifs_info

def add_varmisue_specials(main_file, aux_file, unique_ids, graph):
  with open(main_file, 'r') as mfile:
    _, tok1, row1_s, col1_s, _, _, _, _, _, _, _, _ = get_diff(main_file, aux_file)
    lines = mfile.readlines()
    different_line = lines[row1_s-1]
    new_line = different_line[:col1_s-1] + different_line[col1_s-1:].replace(tok1, CURR_STR, 1)
    lines[row1_s-1] = new_line
    error_location = get_node_by_loc(graph, (row1_s, col1_s-1))
    repair_targets = []
    for ind in unique_ids[tok1]:
      repair_targets.append(get_node_by_loc(graph, ind))
    repair_candidates = []
    for ind in unique_ids:
      for ind2 in unique_ids[ind]:
        node = get_node_by_loc(graph, ind2)
        if hasattr(node, 'ctx') or \
           isinstance(node, ast.arg):
          repair_candidates.append(get_node_by_loc(graph, ind2))
    return error_location, repair_targets, repair_candidates

def get_node_by_loc(G, loc):
  return list({n: d['loc'] for n, d in G.nodes.items() if 'loc' in d and d['loc']==loc}.keys())[0]

def gen_graph_from_source(main_file, aux_file, task_name, pred_kind):
  infile_content = ""
  with open(main_file, 'r') as f:
    infile_content = f.read()
  AST_nodes, assigns, subtrees, ifs_info = get_AST_nodes(infile_content)
  flat_graph = nx.Graph()
  unique_ids = {}
  # ast node : (row, col)->(node, tok)
  for loc in AST_nodes:
    flat_graph.add_node(AST_nodes[loc][0], tok=AST_nodes[loc][1], loc=loc)
    if AST_nodes[loc][1] in unique_ids:
      unique_ids[AST_nodes[loc][1]].append(loc)
    else:
      unique_ids[AST_nodes[loc][1]] = [loc]

  def is_def(checking):
    if isinstance(checking, ast.Subscript):
      if isinstance(checking.ctx, ast.Store) or isinstance(checking.ctx, ast.AugStore):
        if isinstance(checking.value, str):
          return True, checking.value
    if isinstance(checking, ast.Name):
      if isinstance(checking.ctx, ast.Store) or isinstance(checking.ctx, ast.AugStore):
        return True, checking.id
    if isinstance(checking, ast.arg):
      return True, checking.arg
    if isinstance(checking, ast.Attribute):
      return True, checking.value
    return False, None

  # collect definitions
  defs = []
  for item in AST_nodes:
    status, name = is_def(AST_nodes[item][0])
    if status:
      defs.append((AST_nodes[item][0],name))

  # NextToken
  sorted_locs = sorted(AST_nodes.keys())
  for i, loc in enumerate(sorted_locs):
    if i + 1 == len(sorted_locs):
      continue
    prev_node = get_node_by_loc(flat_graph, sorted_locs[i])
    curr_node = get_node_by_loc(flat_graph, sorted_locs[i+1])
    flat_graph.add_edge(prev_node, curr_node, label='enum_NEXT_SYNTAX', id='9')

  # LastLexicalUse
  for n in unique_ids:
    for i in range(1, len(unique_ids[n])):
      prev_node = get_node_by_loc(flat_graph, unique_ids[n][i-1])
      curr_node = get_node_by_loc(flat_graph, unique_ids[n][i])
      flat_graph.add_edge(prev_node, curr_node, label='enum_LAST_LEXICAL_USE', id='10')

  # LastRead/LastWrite
  reads, writes = [], []
  for id in unique_ids:
    for n in unique_ids[id]:
      curr_node = get_node_by_loc(flat_graph, n)
      if isinstance(curr_node, ast.Name):
        if isinstance(curr_node.ctx, ast.Store):
          writes.append(curr_node)
        if isinstance(curr_node.ctx, ast.Load):
          reads.append(curr_node)
  for n in unique_ids:
    for i in range(len(unique_ids[n])):
      for j in range(i, len(unique_ids[n])):
        node1 = get_node_by_loc(flat_graph, unique_ids[n][i])
        node2 = get_node_by_loc(flat_graph, unique_ids[n][j])
        if node2 in writes:
          flat_graph.add_edge(node1, node2, label='enum_LAST_WRITE', id='3')
        if node2 in reads:
          flat_graph.add_edge(node1, node2, label='enum_LAST_READ', id='2')

  # ReturnsTo
  for loc in sorted_locs:
    node = get_node_by_loc(flat_graph, loc)
    if isinstance(node, ast.FunctionDef):
      for elem in node.body:
        if isinstance(elem, ast.Return):
          flat_graph.add_edge(elem, node, label='enum_RETURNS_TO', id='5')

  # ComputedFrom
  for loc in assigns:
    lhs, rhs = [], []
    assignment = assigns[loc][0]
    if isinstance(assignment, ast.Assign):
      lhs += assignment.targets
    if isinstance(assignment, ast.AugAssign):
      lhs += [assignment.target]
    rhs += subtrees[assignment.value]
    for l in lhs:
      for r in rhs:
        flat_graph.add_edge(l, r, label='enum_COMPUTED_FROM', id='4')

  # 'enum_CFG_NEXT' (GuardedBy, GuardedByNeg)
  for if_node in ifs_info:
    for entry1 in ifs_info[if_node]['body']:
      for entry2 in ifs_info[if_node]['test']:
        if entry1.id == entry2.id:
          flat_graph.add_edge(entry1, entry2, label='enum_CFG_NEXT', id='1')

  err_loc = None
  rep_targets = None
  rep_cands = None
  if task_name == 'varmisuse' and pred_kind in ['loc_rep', 'loc_cls']:
    err_loc, rep_targets, rep_cands = add_varmisue_specials(main_file, aux_file, unique_ids, flat_graph)

  return flat_graph, err_loc, rep_targets, rep_cands, defs
