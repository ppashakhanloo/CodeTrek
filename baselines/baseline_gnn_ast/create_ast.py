import ast
from pydot import Edge, Node
from astmonkey import visitors, transformers
from diff import get_diff 

CURR_STR = '___CURR___'
SLOT_STR = 'SlotNode'
IF_IND = 'ast.If'
IF_IND_ = '_ast.If'
NAME_IND = '<ast.Name '
NAME_IND_ = '<_ast.Name '
ARG_IND = '<ast.arg '
ARG_IND_ = '<_ast.arg '
FUNC_IND = '<ast.FunctionDef '
FUNC_IND_ = '<_ast.FunctionDef '
RET_IND = '<ast.Return '
RET_IND_ = '<_ast.Return '
ASSIGN_IND = '<ast.Assign '
ASSIGN_IND_ = '<_ast.Assign '

def build_child_edges(correct_file, incorrect_file):
  _, tok1, row1_s, col1_s, _, _, _, tok2, row2_s, col2_s, _, _ = get_diff(correct_file, incorrect_file)

  with open(correct_file, 'r', 100*(2**20)) as infile:
    lines = infile.readlines()
    different_line = lines[row1_s-1]
    new_line = different_line[:col1_s-1] + different_line[col1_s-1:].replace(tok1, CURR_STR, 1)
    lines[row1_s-1] = new_line
    
    contents = "".join(lines)
    node = ast.parse(contents)
    node = transformers.ParentChildNodeTransformer().visit(node)
    visitor = visitors.GraphNodeVisitor()
    visitor.visit(node)
    graph = visitor.graph
    graph.set_type('digraph')

    neighbors = {} # node -> neighbors
    for edge in graph.get_edges():
      src = edge.get_source()
      dst = edge.get_destination()
      if src not in neighbors.keys():
        neighbors[src] = [dst]
      else:
        neighbors[src].append(dst)
    subtrees = {} # node -> subtree nodes
    for node in neighbors.keys():
      res = []
      get_subtree(node, res, neighbors)
      subtrees[node] = res

    # get if-then-else before renaming the edges
    if_branches = {}
    for node in neighbors.keys():
      if graph.get_node(node)[0].get('label').startswith(IF_IND) or graph.get_node(node)[0].get('label').startswith(IF_IND_):
        condition = ""
        then_branch = []
        else_branch = []
        for neighbor in neighbors[node]:
          if graph.get_edge(node, neighbor)[0].get('label').startswith('test'):
            condition = neighbor
          elif graph.get_edge(node, neighbor)[0].get('label').startswith('body'):
            then_branch.append(neighbor)
          elif graph.get_edge(node, neighbor)[0].get('label').startswith('orelse'):
            else_branch.append(neighbor)
        if_branches[node] = (condition, then_branch, else_branch)

    for edge in graph.get_edges():
      edge.set('label', 'Child')

    # find curr and replace with its variable name
    node_of_interest = ""
    for node in graph.get_nodes():
      if CURR_STR in node.get('label'):
        node.set('label', node.get('label').replace(CURR_STR, tok1))
        node_of_interest = node
        break
    
    return graph, neighbors, subtrees, node_of_interest, if_branches

def add_next_token_edges(graph, subtrees):
  token_nodes = []
  # get leaf nodes
  for node in graph.get_nodes():
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
  if isinstance(node, str):
    return node.startswith(ARG_IND) or node.startswith(NAME_IND) or node.startswith(ARG_IND_) or node.startswith(NAME_IND_)
  return node.get_name().startswith(ARG_IND) or node.get_name().startswith(NAME_IND) or node.get_name().startswith(ARG_IND_) or node.get_name().startswith(NAME_IND_)

def add_last_lexical_use_edges(graph):
  nodes_to_vars = {}
  for node in graph.get_nodes():
    if is_variable_node(node):
      variable = node.obj_dict['attributes']['label'].split("'")[1]
      nodes_to_vars[node] = variable
  
  variables = {}
  for item in nodes_to_vars:
    variables[nodes_to_vars[item]] = []

  for item in nodes_to_vars:
    variables[nodes_to_vars[item]].append(item)

  # variables: {'var': [node1, node2]}
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
    if node.startswith(FUNC_IND) or node.startswith(FUNC_IND_):
      for t in subtrees[node]:
        if t.startswith(RET_IND) or t.startswith(RET_IND_):
          edge = Edge(t, node)
          edge.set('label', 'ReturnsTo')
          graph.add_edge(edge)
  return graph

def add_computed_from_edges(graph, subtrees):
  for node in subtrees:
    if node.startswith(ASSIGN_IND) or node.startswith(ASSIGN_IND_):
      assign_l = subtrees[node][0] # left variable
      assign_r = subtrees[node][1:] # right hand subtree
      lhs_nodes = []
      rhs_nodes = []
      
      if is_variable_node(assign_l):
        lhs_nodes.append(assign_l)
      else:
        lhs_nodes += get_variables(assign_l, subtrees, graph)[0]
     
      for var in assign_r:
        rhs_nodes += get_variables(var, subtrees, graph)[0]
      
      for var1 in lhs_nodes:
        for var2 in rhs_nodes:
          edge = Edge(var2, var1)
          edge.set('label', 'ComputedFrom')
          graph.add_edge(edge)

  return graph

def add_last_read_write_edges(graph, variables):
  variable_writes = {}
  variable_reads = {}
  
  # variables: {'var': [node1, node2]}
  for var in variables:
    for node in variables[var]:
      if 'ast.Store' in node.get('label'):
        if var in variable_writes:
          variable_writes[var].append(node)
        else:
          variable_writes[var] = [node]
      if 'ast.Load' in node.get('label'):
        if var in variable_reads:
          variable_reads[var].append(node)
        else:
          variable_reads[var] = [node]

  for var in variables:
    for node in variables[var]:
      if var in variable_writes:
        for wvar in variable_writes[var]:
          edge = Edge(node, wvar)
          edge.set('label', 'LastWrite')
          graph.add_edge(edge)

  for var in variables:
    for node in variables[var]:
      if var in variable_reads:
        for rvar in variable_reads[var]:
          edge = Edge(node, rvar)
          edge.set('label', 'LastRead')
          graph.add_edge(edge)

  return graph, variable_reads, variable_writes

def get_variables(node, subtrees, graph):
  if node not in subtrees.keys():
    return [], []
  nodes, vars = [], []
  for n in subtrees[node]:
    if is_variable_node(n):
      retrieved_node = graph.get_node(n)[0]
      nodes.append(retrieved_node)
      vars.append(retrieved_node.get('label').split("'")[1])
  return nodes, vars

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

def add_varmisue_specials(graph, node_of_interest):
  # add slot node to specify location
  slot_node = Node(name='SlotNode')
  slot_node.set('label', 'SlotNode')
  graph.add_node(slot_node)
  graph.add_edge(Edge(slot_node, node_of_interest))

  return graph
    
def save_graph(graph, output_file):
  graph.write(output_file+'.dot', format='dot')
  graph.write(output_file+'.pdf', format='pdf')

def get_subtree(node, res, neighbors):
  if node in neighbors:
    for child in neighbors[node]:
      res.append(child)
      get_subtree(child, res, neighbors)

value_indicators = [
  'name=',
  'id=',
  'attr=',
  'arg=',
  'module=',
  'value='
]

value_exclusions = [
  'ast.arguments'
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
  if ind == 'value=':
    return label[loc + offset - 1:label.find(',', loc)]
  return label[loc + offset:label.find('\'', loc + offset)]

def fix_node_labels(graph):
  for node in graph.get_nodes():
    full_label = node.get('label')
    ind = has_value(full_label)
    if ind:
      # create a new node as a terminal node
      terminal_node = Node(name=node.get_name()+'_')
      terminal_node.set('label', 'Terminal' + '[SEP]' + get_value(full_label, ind))
      graph.add_node(terminal_node)
      # add an edge from the non-terminal node to the terminal node
      terminal_edge = Edge(node.get_name(), terminal_node.get_name())
      terminal_edge.set('label', 'Child')
      graph.add_edge(terminal_edge)

    if full_label == 'SlotNode':
      label = full_label
    else:
      label = full_label[4:full_label.find('(')]
    node.set('label', label)

  return graph

def gen_graph_from_source(infile, aux_file):
  graph, neighbors, subtrees, node_of_interest, if_branches = build_child_edges(infile, aux_file)
  graph = add_next_token_edges(graph, subtrees)
  graph, variables = add_last_lexical_use_edges(graph)
  graph = add_returns_to_edges(graph, subtrees)
  graph = add_computed_from_edges(graph, subtrees)
  graph, var_reads, var_writes = add_last_read_write_edges(graph, variables)
  graph = add_guarded_edges(graph, subtrees, if_branches)
  # if the task is var misuse
  graph = add_varmisue_specials(graph, node_of_interest)
  # finally, fix all the labels
  graph = fix_node_labels(graph)
  return graph
