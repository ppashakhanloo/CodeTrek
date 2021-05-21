import os
import sys
import json
import pygraphviz

from data_prep.random_walk.randomwalk import RandomWalker
from data_prep.random_walk.walkutils import WalkUtils
from data_prep.tokenizer import tokenizer

graph_file = sys.argv[1]
stub_file = sys.argv[2]
json_output = sys.argv[3]

graph = RandomWalker.load_graph_from_gv(graph_file)



# filename
# SlotNodeIdx
# ContextGraph
    # Edges
        # EdgeType1 --> list....
        #...
    # NodeTypes
    # NodeValues
    # label


with open(stub_file, 'r') as f:
    obj = json.loads(f.read())
    label = obj[0]['label']
    anchor = obj[0]['anchor']

filename = graph_file
node_types = [0] * (len(graph.nodes()))
node_values = [0] * (len(graph.nodes()))
node_tokens = [0] * (len(graph.nodes()))
anchor_index = 0

for node in graph.nodes(data=True):
    node_index = int(node[0])
    node_label = node[1]['label']
    if node_label == anchor:
        anchor_index = node_index
    node_name, values = WalkUtils.parse_node_label(node_label)
    node_type, node_value = WalkUtils.gen_node_type_value(node_name, values)
    
    if len(node_value) != 0:
        tok = tokenizer.tokenize(node_value, 'python')
        #node_value = tok
    else:
        tok = []

    if node_type.startswith('v_'):
        node_type = 'variable'
    
    node_types[node_index-1] = node_type
    node_values[node_index-1] = str(node_value)
    node_tokens[node_index-1] = tok if len(tok) > 0 else node_type
    print('converted', node_index,';', node_type, ';', node_value)
assert int(anchor_index) > 0

edges = {}
for edge in graph.edges(data=True):
    src = int(edge[0])
    dst = int(edge[1])
    edge_label = edge[2]['label']
    
    if edge_label in edges.keys():
        edges[edge_label].append([src-1, dst-1])
    else:
        edges[edge_label] = [[src-1, dst-1]]

datapoint = {
    "filename": graph_file,
    "SlotNodeIdx": anchor_index,
    "label": label,
    "ContextGraph": {
        "Edges": edges,
        "NodeTypes": node_types,
        "NodeValues": node_values,
        "NodeTokens": node_tokens
    }
}

with open(json_output, 'w') as f:
    f.write(json.dumps(datapoint))
