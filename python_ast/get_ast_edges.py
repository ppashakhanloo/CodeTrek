import ast
import sys
from collections import OrderedDict

input_code_file = sys.argv[1]
output_edges_file = sys.argv[2]

def node_str(node):
    if isinstance(node, ast.AST):
        fields = [(name, node_str(val)) for name, val in ast.iter_fields(node) if name not in ('left', 'right')]
        rv = '%s(%s' % (node.__class__.__name__, ', '.join('%s=%s' % field for field in fields))
        return rv + ')'
    else:
        return repr(node)

def node_kind(node):
  if isinstance(node, ast.AST):
    return node.__class__.__name__

ast_nodes = []
def ast_visit(node, level=0):
    ast_nodes.append((node, level))
    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            for item in value:
                if isinstance(item, ast.AST):
                    ast_visit(item, level=level+1)
        elif isinstance(value, ast.AST):
            ast_visit(value, level=level+1)

def construct_edges():
  edges = []
  prev_node = ast_nodes[0][0]
  prev_level = 0

  for index in range(1, len(ast_nodes)):
    node = ast_nodes[index][0]
    level = ast_nodes[index][1]
    
    if level > prev_level:
      edges.append((prev_node, node))
      prev_node = node
      prev_level = level
    elif level <= prev_level:
      for ind in reversed(range(index)):
        if ast_nodes[ind][1] < level:
          edges.append((ast_nodes[ind][0], node))
          prev_node = node
          prev_level = level
          break

  return edges


with open(input_code_file, 'r') as infile:
  code = infile.readlines()
  code = ''.join(code)
  ast_visit(ast.parse(code))

  node_ids = dict()
  index = 0
  for node in ast_nodes:
    node_ids[node[0]] = index
    index += 1
  
  with open(output_edges_file, 'w') as outfile:
    for item in construct_edges():
      outfile.write(str(node_ids[item[0]]) + " " + str(node_ids[item[1]]) + "\n")
