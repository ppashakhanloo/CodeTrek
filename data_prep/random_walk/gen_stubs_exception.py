import sys
import csv
import json
import os
from pygraphviz import Node
from data_prep.random_walk.datapoint import DataPoint, TrajNodeValue
from data_prep.random_walk.walkutils import WalkUtils
from data_prep.random_walk.randomwalk import RandomWalker
from typing import List



def main(args: List[str]) -> None:
    gv_file = args[1]
    edb_path = args[2]
    gt = args[3]
    out_file = args[4]
    walks_or_graphs = args[5]

    gt_exception = []
    anchor_node = []
    holes = []

    gt_exception.append(gt.strip())
    
    assert len(gt_exception) == 1

    with open(os.path.join(edb_path, "except_hole.bqrs.csv"), 'r') as exceptions_file:
        reader = csv.reader(exceptions_file, delimiter=',')
        rows = list(reader)
        for row in rows[1:]:
            holes.append(row)

    assert len(holes) == 1

    graph = RandomWalker.load_graph_from_gv(gv_file)
    for node in graph.nodes():
        element = graph.nodes[node]['label']
        relname = element[:element.find('(')]
        cols = element[element.find('(')+1:element.find(')')].split(',')
        if relname == 'py_stmts' and cols in holes:
            label = gt_exception[len(anchor_node)-1]
            anchor_node.append((node, label))
    
    walklist = []
    for anchor, label in anchor_node:
        anchor_label = graph.nodes[anchor]['label']
        if walks_or_graphs == 'graphs':
            walks = []
        else:
            random_walk = RandomWalker(graph, anchor_label, 'python')
            walks = random_walk.random_walk(max_num_walks=100, min_num_steps=8, max_num_steps=30)
        # generate the Json file
        source = gv_file
        traj_anchor = TrajNodeValue(anchor_label)
        trajectories = [WalkUtils.build_trajectory(walk) for walk in walks]
        data_point = DataPoint(traj_anchor, trajectories, [], label, source)
        walklist.append(data_point.to_dict())
    js = json.dumps(walklist)
    
    with open(out_file, 'w') as op_file:
        op_file.write(js)

if __name__ == '__main__':
    main(sys.argv)
