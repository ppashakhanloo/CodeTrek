import ast

import asttokens
import networkx as nx

from diff import get_diff

CURR_STR = 'HOLE'

def get_AST_nodes(contents):
  nodes_AST = {}
  assigns = {}
  subtrees = {}
  ifs_info = {}

  atok = asttokens.ASTTokens(contents, parse=True)
  for node in ast.walk(atok.tree):
    if hasattr(node, 'lineno') and hasattr(node, 'id'):
      nodes_AST[(node.lineno, node.col_offset)] = (node, node.id)
    elif hasattr(node, 'lineno') and hasattr(node, 'name'):
      nodes_AST[(node.lineno, node.col_offset)] = (node, node.name)
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

    if isinstance(node, ast.AugAssign):
      assigns[(node.lineno, node.col_offset)] = (node, 'assign')
      subtrees[node.value] = []
    elif isinstance(node, ast.Assign):
      assigns[(node.lineno, node.col_offset)] = (node, 'assign')
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

def add_defuse_specials(label, graph):
  classes = ['used', 'unused']
  error_location = get_node_by_loc(graph, (0, 0))
  repair_targets = []
  repair_candidates = []
  return error_location, repair_targets, repair_candidates, classes.index(label)

def add_exception_specials(unique_ids, corr_except, graph):
  classes = ["ValueError", "KeyError", "AttributeError", "TypeError",
             "OSError", "IOError", "ImportError", "IndexError", "DoesNotExist",
             "KeyboardInterrupt", "StopIteration", "AssertionError", "SystemExit",
             "RuntimeError", "HTTPError", "UnicodeDecodeError", "NotImplementedError",
             "ValidationError", "ObjectDoesNotExist", "NameError"]
  error_location = get_node_by_loc(graph, unique_ids['HoleException'][0])
  repair_targets = []
  repair_candidates = []
  return error_location, repair_targets, repair_candidates, classes.index(corr_except)

def add_varmisue_specials(main_file, aux_file, unique_ids, graph, vm_label):
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
        repair_candidates.append(get_node_by_loc(graph, ind2))
    return error_location, repair_targets, repair_candidates, vm_label

def get_node_by_loc(G, loc):
  return list({n: d['loc'] for n, d in G.nodes.items() if 'loc' in d and d['loc']==loc}.keys())[0]

def gen_graph_from_source(main_file, aux_file, task_name, corr_except=None, defuse_label=None, vm_label=None):
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
  flat_graph.add_node(ast.Expr(), tok='', loc=(0,0))

  # NextToken
  sorted_locs = sorted(AST_nodes.keys())
  for i, loc in enumerate(sorted_locs):
    if i + 1 == len(sorted_locs):
      continue
    prev_node = get_node_by_loc(flat_graph, sorted_locs[i])
    curr_node = get_node_by_loc(flat_graph, sorted_locs[i+1])
    flat_graph.add_edge(prev_node, curr_node, label='enum_NEXT_SYNTAX', id='8')

  # LastLexicalUse
  for n in unique_ids:
    for i in range(1, len(unique_ids[n])):
      prev_node = get_node_by_loc(flat_graph, unique_ids[n][i-1])
      curr_node = get_node_by_loc(flat_graph, unique_ids[n][i])
      flat_graph.add_edge(prev_node, curr_node, label='enum_LAST_LEXICAL_USE', id='9')

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
          flat_graph.add_edge(node1, node2, label='enum_LAST_WRITE', id='2')
        if node2 in reads:
          flat_graph.add_edge(node1, node2, label='enum_LAST_READ', id='1')

  # ReturnsTo
  for loc in sorted_locs:
    node = get_node_by_loc(flat_graph, loc)
    if isinstance(node, ast.FunctionDef):
      for elem in node.body:
        if isinstance(elem, ast.Return):
          flat_graph.add_edge(elem, node, label='enum_RETURNS_TO', id='4')

  # ComputedFrom
  for loc in assigns:
    lhs, rhs = [], []
    assignment = assigns[loc][0]
    if isinstance(assignment, ast.Assign):
      lhs += assignment.targets
    elif isinstance(assignment, ast.AugAssign):
      lhs += [assignment.target]
    rhs += subtrees[assignment.value]
    for l in lhs:
      for r in rhs:
        flat_graph.add_edge(l, r, label='enum_COMPUTED_FROM', id='3')

  # 'enum_CFG_NEXT' (GuardedBy, GuardedByNeg)
  for if_node in ifs_info:
    for entry1 in ifs_info[if_node]['body']:
      for entry2 in ifs_info[if_node]['test']:
        if entry1.id == entry2.id:
          flat_graph.add_edge(entry1, entry2, label='enum_CFG_NEXT', id='0')

  if task_name == 'varmisuse':
    err_loc, rep_targets, rep_cands, label = add_varmisue_specials(main_file, aux_file, unique_ids, flat_graph, vm_label)
  elif task_name == 'defuse':
    err_loc, rep_targets, rep_cands, label = add_defuse_specials(defuse_label, flat_graph)
  elif task_name == 'exception':
    err_loc, rep_targets, rep_cands, label = add_exception_specials(unique_ids, corr_except, flat_graph)
  else:
    raise ValueError(task_name, 'not a valid task name.')

  return flat_graph, err_loc, rep_targets, rep_cands, label
