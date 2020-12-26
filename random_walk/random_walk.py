import sys
import networkx as nx
from pygraphviz import AGraph
from random import choice

# The output is a list of walks starting from the specified node. Each walk is
# in this format: [(n1, n1_lbl), (n1, n2, e1_lbl), (n2, n2_lbl), ...]
#                  _____^______  _______^_______   _____^_____
#                     node            edge             node
#
def random_walk(graph, node, num_walks=100, num_steps=3,
                   include_nodes=None):
  walks = list()
  for walk in range(num_walks):
    curr_walk = [(node, graph.nodes[node]['label'])]
    curr_node = node
    curr_edge = None
    for step in range(num_steps):
      neighbors = list(graph.neighbors(curr_node))
      if len(neighbors) > 0:
        prev_node = curr_node
        curr_node = choice(neighbors)
        curr_edge = (prev_node, curr_node,
            choice(list(graph[prev_node][curr_node].values()))['label'])
      
      curr_walk.append(curr_edge)
      curr_walk.append((curr_node, graph.nodes[curr_node]['label']))
    
    # remove cyclic walks
    if not any(curr_walk.count(n) > 1 for n in curr_walk):
      # include nodes
      if not include_nodes:
        walks.append(curr_walk)
      else:
        if all(item in curr_walk for item in include_nodes):
          walks.append(curr_walk)

  return walks

def load_graph_from_gv(path):
  graph = AGraph(path, directed=False)
  return nx.nx_agraph.from_agraph(graph)


if __name__ == '__main__':
  gv_file = sys.argv[1]
  graph = load_graph_from_gv(gv_file)
  for node in graph.nodes():
    print(random_walk(graph, node, num_walks=10, num_steps=3))
