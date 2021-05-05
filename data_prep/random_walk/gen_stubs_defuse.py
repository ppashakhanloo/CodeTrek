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
    if not len(args) == 5:
        print('Usage: python3 gen_walks.py <gv-file> <edb_path> <ground-truth> <out-file>')
        exit(1)

    gv_file = args[1]
    edb_path = args[2]
    gt_path = args[3]
    out_file = args[4]

    anchor_nodes = []
    gt_vars = set()
    local_variable_rows = []

    with open(gt_path, 'r') as gt_file:
        reader = csv.reader(gt_file, delimiter=',')
        rows = list(reader)
        for gt in rows[1:]:
            gt_vars.add(gt[1])
    
    with open(os.path.join(edb_path, "local_variable.bqrs.csv"), 'r') as local_variable_file:
        reader = csv.reader(local_variable_file, delimiter=',')
        rows = list(reader)
        for row in rows[1:]:
            local_variable_rows.append(row)

    graph = RandomWalker.load_graph_from_gv(gv_file)
    for node in graph.nodes():
        element = graph.nodes[node]['label']
        relname = element[:element.find('(')]
        cols = element[element.find('(')+1:element.find(')')].split(',')
        if relname == 'variable' and cols in local_variable_rows:
            label = 'used'
            if cols[0] in gt_vars:
                label = 'unused'
            anchor_nodes.append((node, label))
    
    walklist = []
    for anchor, label in anchor_nodes:
        anchor_label = graph.nodes[anchor]['label']
        walks = []
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
