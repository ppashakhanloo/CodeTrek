import sys
import ast
from pydot import Edge, Node, Graph
from astmonkey import visitors, transformers
from diff import get_diff 

CURR_STR = '___CURR___'

def build_Child_graph(correct_file, incorrect_file):
  _, tok1, row1_s, col1_s, row1_e, col1_e, _, tok2, row2_s, col2_s, row2_e, col2_e = get_diff(correct_file, incorrect_file)
  
  lines = []
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
    
    for edge in graph.get_edges():
      edge.set('label', 'Child')
    
    neighbors = {} # node -> neighbors
    for edge in graph.get_edges():
      src = edge.get_source()
      dst = edge.get_destination()
      if src not in neighbors:
        neighbors[src] = [dst]
      else:
        neighbors[src].append(dst)
    
    subtrees = {} # node -> subtree nodes
    for node in neighbors:
      res = []
      get_subtree(node, res, neighbors)
      subtrees[node] = res
   
    # find curr and replace with its variable name
    node_of_interest = ""
    for node in graph.get_nodes():
      if CURR_STR in node.get('label'):
        node.set('label', node.get('label').replace(CURR_STR, tok1))
        node_of_interest = node
        break
    
    return graph, neighbors, subtrees, node_of_interest
  return None, None, None, None

def add_NextToken_edges(graph, subtrees):
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
    return node.startswith('<_ast.arg ') or node.startswith('<_ast.Name ')
  return node.get_name().startswith('<_ast.arg ') or node.get_name().startswith('<_ast.Name ')

def add_LastLexicalUse_edges(graph):
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

def add_ReturnsTo_edges(graph, subtrees):
  for node in subtrees:
    if node.startswith("<_ast.FunctionDef "):
      for t in subtrees[node]:
        if t.startswith("<_ast.Return "):
          edge = Edge(t, node)
          edge.set('label', 'ReturnsTo')
          graph.add_edge(edge)
  return graph

def add_ComputedFrom_edges(graph, subtrees):
  for node in subtrees:
    if node.startswith("<_ast.Assign "):
      assign_l = subtrees[node][0] # left variable
      assign_r = subtrees[node][1:] # right hand subtree
      lhs_nodes = []
      rhs_nodes = []
      
      if is_variable_node(assign_l):
        lhs_nodes.append(assign_l)
      else:
        lhs_nodes += variables(assign_l, subtrees, graph)[0]
     
      for var in assign_r:
        rhs_nodes += variables(var, subtrees, graph)[0]
      
      for var1 in lhs_nodes:
        for var2 in rhs_nodes:
          edge = Edge(var2, var1)
          edge.set('label', 'ComputedFrom')
          graph.add_edge(edge)

  return graph

def add_LastReadWrite_edges(graph, subtrees, variables):
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

def variables(node, subtrees, graph):
  res = []
  names = []
  if node not in subtrees:
    return res, names
  for n in subtrees[node]:
    if is_variable_node(n):
      res.append(n)
      names.append(graph.get_node(n)[0].obj_dict['attributes']['label'].split("'")[1])
  return res, names

def add_Guarded_edges(graph, subtrees, neighbors):
  for node in subtrees:
    if node.startswith("<_ast.If "):
      guard_node = neighbors[node][0]
      guard_vars, guard_var_names = variables(neighbors[node][0], subtrees, graph)
      then_vars, then_var_names = variables(neighbors[node][1], subtrees, graph)
      else_vars, else_var_names = [], []
      if len(neighbors[node]) == 3:
        else_vars, else_var_names = variables(neighbors[node][2], subtrees, graph)
      
      for i in range(len(then_var_names)):
        if then_var_names[i] in guard_var_names:
          edge = Edge(then_vars[i], guard_node)
          edge.set('label', 'GuardedBy')
          graph.add_edge(edge)

      for i in range(len(else_var_names)):
        if else_var_names[i] in guard_var_names:
          edge = Edge(else_vars[i], guard_node)
          edge.set('label', 'GuardedByNegation')
          graph.add_edge(edge)

  return graph

def add_varmisue_specials(graph, variables, node_of_interest):
  # add slot node to specify location
  slot_node = Node(name='<SLOT>')
  slot_node.set('label', node_of_interest.get_name())
  graph.add_node(slot_node)
  graph.add_edge(Edge(slot_node, node_of_interest))

  return graph
    
def save_graph(graph, output_file):
  graph.write(output_file+'.dot', format='dot')
  graph.write(output_file+'.png', format='png')

def get_subtree(node, res, neighbors):
  if node in neighbors:
    for child in neighbors[node]:
      res.append(child)
      get_subtree(child, res, neighbors)

def gen_graph_from_source(infile, aux_file):
  graph, neighbors, subtrees, node_of_interest = build_Child_graph(infile, aux_file)
  graph = add_NextToken_edges(graph, subtrees)
  graph, variables = add_LastLexicalUse_edges(graph)
  graph = add_ReturnsTo_edges(graph, subtrees)
  graph = add_ComputedFrom_edges(graph, subtrees)
  graph, var_reads, var_writes = add_LastReadWrite_edges(graph, subtrees, variables)
  graph = add_Guarded_edges(graph, subtrees, neighbors)
  # if the task is var misuse
  graph = add_varmisue_specials(graph, variables, node_of_interest)
  return graph

def main(args):
  graph = gen_graph_from_source(args[1], args[2])

if __name__ == "__main__":
  main(sys.argv)
