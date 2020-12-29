import sys
import networkx as nx
from pygraphviz import AGraph
from random import choice, randint


# The output is a list of simple walks starting from the specified node.
# Each walk: [(n1, n1_lbl), (n1, n2, e1_lbl), (n2, n2_lbl), ...]
#             _____^______  _______^_______   _____^_____
#                node            edge             node
# The length of walks can be different and is specified by
# min_num_steps and max_num_steps.
# The number of random walks can be less than max_num_walks if there are less
# walks from the source than the requested number of walks.
#
def random_walk(graph, source, max_num_walks, min_num_steps, max_num_steps):
  walks = list()
  outer_loop = 0
  walk = 0
  while walk < max_num_walks*3 and len(walks) < max_num_walks:
    walk += 1
    curr_walk = [(source, graph.nodes[source]['label'])]
    curr_node = source
    curr_edge = None
    random_num_steps = randint(min_num_steps, max_num_steps)
    
    for step in range(random_num_steps):
      neighbors = list(graph.neighbors(curr_node))
      if len(neighbors) > 0:
        prev_node = curr_node
        curr_node = choice(neighbors)
        curr_edge = (prev_node, curr_node,
            choice(list(graph[prev_node][curr_node].values()))['label'])
        curr_edge_rev = (curr_edge[1], curr_edge[0], curr_edge[2])
      
        if curr_edge not in curr_walk \
           and curr_edge_rev not in curr_walk \
           and (curr_node, graph.nodes[curr_node]['label']) not in curr_walk:
          curr_walk.append(curr_edge)
          curr_walk.append((curr_node, graph.nodes[curr_node]['label']))
        else:
          curr_node = prev_node
    
    if curr_walk not in walks and len(curr_walk) > 1:
      walks.append(curr_walk)

  return walks

# find __all__ simple paths in the graph starting from the specified node,
# with a maximum length specified by cutoff
def simple_walks(graph, node, cutoff):
  walks = list()
  for target in graph.nodes():
    all_paths = nx.algorithms.all_simple_paths(graph, node, target, cutoff)
    for path in map(nx.utils.pairwise, all_paths):
      walks.append(list(path))

  return walks

def load_graph_from_gv(path):
  graph = AGraph(path, directed=False)
  return nx.nx_agraph.from_agraph(graph)
