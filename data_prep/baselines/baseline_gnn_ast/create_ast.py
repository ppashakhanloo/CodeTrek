import ast
import copy

from data_prep.baselines.baseline_gnn_ast.diff import get_diff
from pydot import Edge, Node
from astmonkey import visitors, transformers

CURR_STR = '___CURR___'
FUNC = 'ast.FunctionDef'
RETURN = 'ast.Return'
ASSIGN = 'ast.Assign'
NAME = 'ast.Name'
ARG = 'ast.arg'
ARGS = 'ast.arguments'
STORE = 'ast.Store'
LOAD = 'ast.Load'
IF = 'ast.If'

def get_graph(contents):
  node = ast.parse(contents)
  node_positions = dict()
  for n in ast.walk(node):
    if hasattr(n, 'lineno') and hasattr(n, 'col_offset'):
      node_positions[str(n)] = [n.lineno, n.col_offset]
  node = transformers.ParentChildNodeTransformer().visit(node)
  visitor = visitors.GraphNodeVisitor()
  visitor.visit(node)
  graph = visitor.graph
  graph.set_type('digraph')
  return graph, node_positions

def build_child_edges(main_file, aux_file, task_name, pred_kind):
  node_of_interest = None
  neighbors = {} # node -> neighbors
  subtrees = {} # node -> subtree nodes
  if_branches = {}
  tok1 = None
  tok2 = None

  with open(main_file, 'r') as infile:
    if task_name == 'varmisuse':
      _, tok1, row1_s, col1_s, _, _, _, tok2, row2_s, col2_s, _, _ = get_diff(main_file, aux_file)
      lines = infile.readlines()
      different_line = lines[row1_s - 1]
      new_line = different_line[:col1_s - 1] + different_line[col1_s - 1:].replace(tok1, CURR_STR, 1)
      lines[row1_s - 1] = new_line
      contents = "".join(lines)
      graph, node_positions = get_graph(contents)
      index = 0
      nodes = graph.get_nodes()
      for i in range(len(nodes)):
        if CURR_STR in nodes[i].get('label'):
          index = i
          break
      nodes[index].set('label', nodes[index].get('label').replace(CURR_STR, tok1))
      node_of_interest = nodes[index]
    else:
      graph, node_positions = get_graph(infile.read())

    # compute the neighbors of each node
    for edge in graph.get_edges():
      src = edge.get_source()
      dst = edge.get_destination()
      if src not in neighbors.keys():
        neighbors[src] = [dst]
      else:
        if dst not in neighbors[src]: neighbors[src].append(dst)
      if dst not in neighbors.keys():
        neighbors[dst] = [src]
      else:
        if src not in neighbors[dst]: neighbors[dst].append(src)

    # compute the flat subtree for each node
    neighbor_keys = neighbors.keys()
    for node in neighbor_keys:
      res = []
      get_subtree(node, res, neighbors)
      subtrees[node] = res

    # compute if-then-else information before renaming the edges
    for node in neighbor_keys:
      if IF in graph.get_node(node)[0].get('label'):
        condition = ""
        then_branch = []
        else_branch = []
        for neighbor in neighbors[node]:
          if len(graph.get_edge(node, neighbor)) < 1:
              continue
          if graph.get_edge(node, neighbor)[0].get('label').startswith('test'):
            condition = neighbor
          if graph.get_edge(node, neighbor)[0].get('label').startswith('body'):
            then_branch.append(neighbor)
          if graph.get_edge(node, neighbor)[0].get('label').startswith('orelse'):
            else_branch.append(neighbor)
        if_branches[node] = (condition, then_branch, else_branch)

    for edge in graph.get_edges():
      edge.set('label', 'Child')
  return graph, neighbors, subtrees, copy.deepcopy(node_of_interest), if_branches, tok1, tok2, node_positions

def add_next_token_edges(graph, subtrees):
  token_nodes = []
  # get leaf nodes
  nodes = graph.get_nodes()
  for node in nodes:
    if node not in subtrees:
      token_nodes.append(node)
  # add token edges
  first_node = token_nodes[0]
  for node in token_nodes[1:]:
    edge = Edge(first_node, node)
    edge.set('label', 'NextToken')
    graph.add_edge(edge)
    first_node = node
  return graph

def is_variable_node(node):
  return (NAME in str(node) or ARG in str(node)) and (ARGS not in str(node))

def add_last_lexical_use_edges(graph):
  nodes_to_vars = {}
  nodes = graph.get_nodes()
  for node in nodes:
    if is_variable_node(node):
      variable = node.obj_dict['attributes']['label'].split("'")[1]
      nodes_to_vars[node] = variable
  variables = {}
  for item in nodes_to_vars:
    variables[nodes_to_vars[item]] = []
  for item in nodes_to_vars:
    variables[nodes_to_vars[item]].append(item)
  for v in variables:
    first_node = variables[v][0]
    if len(variables[v]) > 1:
      for node in variables[v][1:]:
        edge = Edge(first_node, node)
        edge.set('label', 'LastLexicalUse')
        graph.add_edge(edge)
        first_node = node
  return graph, variables

def add_returns_to_edges(graph, subtrees):
  for node in subtrees:
    if FUNC in node:
      for t in subtrees[node]:
        if RETURN in t:
          edge = Edge(t, node)
          edge.set('label', 'ReturnsTo')
          graph.add_edge(edge)
  return graph

def add_computed_from_edges(graph, subtrees, neighbors):
  for node in subtrees:
    if ASSIGN in node:
      assign_l = neighbors[node][-2] # left variable
      assign_r = subtrees[node][-1] # right hand subtree (list)
      lhs_node = graph.get_node(assign_l)[0]
      rhs_nodes, _ = get_variables(assign_r, subtrees, graph)
      for r in rhs_nodes:
        edge = Edge(lhs_node, r)
        edge.set('label', 'ComputedFrom')
        graph.add_edge(edge)
  return graph

def add_last_read_write_edges(graph, variables):
  variable_writes = {}
  variable_reads = {}

  # variables: {'var': [node1, node2, ..]}
  var_order = []
  for var in variables:
    for node in variables[var]:
      var_order.append(node)
      if STORE in node.get('label'):
        if var in variable_writes:
          variable_writes[var].append(node)
        else:
          variable_writes[var] = [node]
      if LOAD in node.get('label'):
        if var in variable_reads:
          variable_reads[var].append(node)
        else:
          variable_reads[var] = [node]

  def get_last_read(var_node, var, i_in_var_order):
    list_prefix = var_order[:i_in_var_order]
    for n in reversed(list_prefix):
      v = n.get('label').split("'")[1]
      if v == var:
        if var in variable_reads.keys() and n in variable_reads[var]:
          return n
    return None

  def get_last_write(var_node, var, i_in_var_order):
    list_prefix = var_order[:i_in_var_order]
    for n in reversed(list_prefix):
      v = n.get('label').split("'")[1]
      if v == var:
        if var in variable_writes.keys() and n in variable_writes[var]:
          return n
    return None

  for i in range(1, len(var_order)):
    v = var_order[i].get('label').split("'")[1]
    node_r = get_last_read(var_order[i], v, i)
    node_w = get_last_write(var_order[i], v, i)

    if node_r:
      edge = Edge(var_order[i], node_r)
      edge.set('label', 'LastRead')
      graph.add_edge(edge)
    if node_w:
      edge = Edge(var_order[i], node_w)
      edge.set('label', 'LastWrite')
      graph.add_edge(edge)

  return graph

def get_variables(node, subtrees, graph):
  if node not in subtrees.keys():
    return [], []
  nodes, vs = [], []
  for n in subtrees[node]:
    if is_variable_node(n):
      retrieved_node = graph.get_node(n)[0]
      nodes.append(retrieved_node)
      vs.append(retrieved_node.get('label').split("'")[1])
  return nodes, vs

def add_guarded_edges(graph, subtrees, if_branches):
  for if_node in if_branches.keys():
    guard = if_branches[if_node][0]
    guard_nodes, guard_vars = get_variables(guard, subtrees, graph)

    then_nodes, then_vars = [], []
    for node in if_branches[if_node][1]:
      n, v = get_variables(node, subtrees, graph)
      then_nodes += n
      then_vars += v

    else_nodes, else_vars = [], []
    for node in if_branches[if_node][2]:
      n, v = get_variables(node, subtrees, graph)
      else_nodes += n
      else_vars += v

    for i in range(len(then_vars)):
      if then_vars[i] in guard_vars:
        edge = Edge(then_nodes[i], guard)
        edge.set('label', 'GuardedBy')
        graph.add_edge(edge)

    for i in range(len(else_vars)):
      if else_vars[i] in guard_vars:
        edge = Edge(else_nodes[i], guard)
        edge.set('label', 'GuardedByNegation')
        graph.add_edge(edge)

  return graph

def save_graph(graph, output_file):
  graph.write(output_file+'.pdf', format='pdf')

def get_subtree(node, res, neighbors, i=0):
  if i == 2: return
  for child in neighbors[node]:
    if child in res:
        continue
    res.append(child)
    get_subtree(child, res, neighbors, i + 1)

value_indicators = [
  'name=',
  'id=',
  'attr=',
  'arg=',
  'module=',
  'value='
]

value_exclusions = [
  'ast.arguments',
  'ast.Str'
]

def has_value(label):
  for ind in value_exclusions:
    if ind in label:
      return None
  for ind in value_indicators:
    if ind in label:
      return ind
  return None

def get_value(label, ind):
  loc = label.find(ind)
  offset = len(ind) + 1
  if ind == 'value=' or ind == 'id':
    return label[loc + offset - 1:label.find(',', loc)]
  if ind == 'attr=':
    return label[loc + offset:label.find(',', loc)-1]
  return label[loc + offset:label.find('\'', loc + offset)]

def fix_node_labels(graph):
  nodes = graph.get_nodes()
  terminal_vars = []
  terminal_dus = []
  hole_exception = None
  for node in nodes:
    full_label = node.get('label')
    ind = has_value(full_label)
    cut_label = full_label[4:full_label.find('(')]
    if ind:
      # create a new node as a terminal node
      terminal_node = Node(name=node.get_name()+'_')
      terminal_node.set('label', 'Terminal' + '[SEP]' + get_value(full_label, ind))
      if get_value(full_label, ind) == 'HoleException':
        hole_exception = terminal_node
      graph.add_node(terminal_node)
      # add an edge from the non-terminal node to the terminal node
      terminal_edge = Edge(node.get_name(), terminal_node.get_name())
      terminal_edge.set('label', 'Child')
      graph.add_edge(terminal_edge)
      if cut_label == 'arg' or cut_label == 'Name' or cut_label == 'Attribute':
        if 'ctx=ast.Load' in full_label or 'ctx=ast.AugLoad' in full_label:
          terminal_vars.append((terminal_node, get_value(full_label, ind), 'read'))
        else:
          terminal_vars.append((terminal_node, get_value(full_label, ind), 'write'))
      if get_value(full_label, ind) == 'HoleException':
        hole_exception = terminal_node
    node.set('label', cut_label)
  return graph, terminal_vars, hole_exception

def gen_graph_from_source(infile, aux_file, task_name, pred_kind='prog_cls'):
  graph, neighbors, subtrees, node_of_interest, if_branches, tok1, tok2, _ = build_child_edges(infile, aux_file, task_name, pred_kind)
  graph = add_next_token_edges(graph, subtrees)
  graph, variables = add_last_lexical_use_edges(graph)
  graph = add_returns_to_edges(graph, subtrees)
  graph = add_computed_from_edges(graph, subtrees, neighbors)
  graph = add_last_read_write_edges(graph, variables)
  graph = add_guarded_edges(graph, subtrees, if_branches)
  graph, terminal_vars, hole_exception = fix_node_labels(graph)
  return graph, terminal_vars, node_of_interest, hole_exception, tok1, tok2
