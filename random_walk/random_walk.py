import sys
import networkx as nx
from pygraphviz import AGraph
from random import choice

def random_walk(graph, node, num_walks=100, num_steps=3,
                   include_nodes=None):
  walks = list()
  for walk in range(num_walks):
    curr_walk = [node]
    curr = node
    for step in range(num_steps):
      neighbors = list(graph.neighbors(curr))
      if len(neighbors) > 0:
        curr = choice(neighbors)
      curr_walk.append(curr)
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
    print(random_walk(graph, node, num_walks=10, num_steps=4, include_nodes=['n2']))
