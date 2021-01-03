import sys
import csv
import json
import networkx as nx
from pygraphviz import AGraph
from random import choice
from random_walk import random_walk
from data_prep.datapoint import DataPoint, TrajNode
from data_prep.walkutils import WalkUtils
from typing import List


def load_graph_from_gv(path: str) -> AGraph:
    graph = AGraph(path, directed=False)
    return nx.nx_agraph.from_agraph(graph)


def main(args: List[str]) -> None:
    if not len(args) == 5:
        print('Usage: python3 gen_walks.py <gv-file> <anchor-rel> <ground-truth> <out-file>')
        exit(1)

    gv_file = args[1]
    anchor_rel = args[2]
    gt_path = args[3]
    out_file = args[4]

    gt_vars = set()
    anchor_nodes = set()

    with open(gt_path, 'r') as gt_file:
        reader = csv.reader(gt_file, delimiter=',')
        rows = list(reader)
        for gt in rows[1:]:
            gt_vars.add(gt[1])
    
    graph = load_graph_from_gv(gv_file)
    for node in graph.nodes():
        element = graph.nodes[node]['label']
        relname = element[:element.find('(')]
        cols = element[element.find('(')+1:element.find(')')].split(',')
        if relname == anchor_rel:
            label = 'used'
            if cols[0] in gt_vars:
                label = 'unused'
            anchor_nodes.add((node, label))
    
    # for node, label in anchor_nodes:
    #     print(node, graph.nodes[node]['label'], label)

    walklist = []
    for anchor, label in anchor_nodes:
        walks = random_walk(graph, anchor, max_num_walks=10, min_num_steps=6, max_num_steps=15)
        # generate the Json file
        source = gv_file
        anchor = TrajNode(graph.nodes[anchor]['label'])
        trajectories = [WalkUtils.parse_trajectory(walk) for walk in walks]
        # TODO: use the real hints
        hints = []
        data_point = DataPoint(anchor, trajectories, hints, label, source)
        walklist.append(data_point.to_dict())
    # data_point.dump_json(out_file)
    js = json.dumps(walklist, indent=4)
    
    with open(out_file, 'w') as op_file:
        op_file.write(js)
        print("Written the walks to " + out_file)


if __name__ == '__main__':
    main(sys.argv)
