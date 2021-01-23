import sys
import csv
import json
import os
from pygraphviz import Node
from data_prep.datapoint import DataPoint, TrajNode
from data_prep.hintutils import HintUtils
from data_prep.walkutils import WalkUtils
from randomwalk import RandomWalker
from typing import List


def used_var_hint(walk: List, var_node: Node) -> str:
    return 'pos' if HintUtils.is_used_local_var(walk, var_node) else 'neg'

def get_anchor(edb_dir: str, label: str):
    misuse_locs = []
    with open(os.path.join(edb_dir, "var_misuses.csv")) as misuse_file:
        reader = csv.reader(misuse_file, delimiter=',')
        for row in reader:
            if label == "correct":
                loc = [ x.strip() for x in row[:5] ] + [str(int(row[5])-1)]
                if loc[1][0] == "@":
                    loc[3] = str(int(loc[3]) + 1)
                if loc not in misuse_locs:
                    misuse_locs.append(loc)
            else:
                assert label == "misuse"
                loc = [ x.strip() for x in row[6:-1] ] + [str(int(row[-1])-1)]
                if loc[1][0] == "@":
                    loc[3] = str(int(loc[3]) + 1)
                if loc not in misuse_locs:
                    misuse_locs.append(loc)
    
    print(misuse_locs)
    assert len(misuse_locs) == 1
    misuse_loc_ids = []

    with open(os.path.join(edb_dir, "locations_ast.csv")) as locations_ast,\
        open(os.path.join(edb_dir, "py_locations.csv")) as py_locations:
        ast_loc_reader = csv.reader(locations_ast, delimiter=',')
        ast_loc_rows = list(ast_loc_reader)
        ast_loc_ids = []
        for loc in misuse_locs:
            for row in ast_loc_rows:
                if loc[2:] == row[2:]:
                    ast_loc_ids.append(row)
        print(ast_loc_ids)
        py_loc_reader = csv.reader(py_locations, delimiter=',')
        py_loc_rows = list(py_loc_reader)
        for loc in ast_loc_ids:
            for row in py_loc_rows:
                if loc[0] == row[0]:
                    misuse_loc_ids.append(row)

    assert len(misuse_loc_ids) >= 1

    anchors = set()
    with open(os.path.join(edb_dir, "py_variables.csv")) as py_variables:
        reader = csv.reader(py_variables, delimiter=',')
        for row in reader:
            for id in misuse_loc_ids:
                if row[1] == id[1]:
                    anchors.add('py_variables(' + ','.join(row) + ')')
    
    assert len(anchors) >= 1
    return anchors


def main(args: List[str]) -> None:
    if not len(args) == 5:
        print('Usage: python3 gen_walks.py <gv-file> <edb_path> <ground-truth> <out-file>')
        exit(1)

    gv_file = args[1]
    edb_path = args[2]
    gt = args[3]
    assert gt in ("misuse", "correct")
    out_file = args[4]

    anchor_nodes = set()

    print("Loading Anchors")
    anchors = get_anchor(edb_path, gt)
    print("Anchors Loaded")

    print("Loading Graph")
    graph = RandomWalker.load_graph_from_gv(gv_file)
    print("Graph Loaded")


    for node in graph.nodes():
        element = graph.nodes[node]['label']
        if element in anchors:
            anchor_nodes.add((node, gt))
    
    for node, label in anchor_nodes:
        print(node, graph.nodes[node]['label'], label)

    walklist = []
    for anchor, label in anchor_nodes:
        anchor_label = graph.nodes[anchor]['label']
        walks = RandomWalker.random_walk(graph, anchor, max_num_walks=100, min_num_steps=8, max_num_steps=16)
        # generate the Json file
        source = gv_file
        traj_anchor = TrajNode(anchor_label)
        trajectories = [WalkUtils.build_trajectory(walk) for walk in walks]
        hints = []
        data_point = DataPoint(traj_anchor, trajectories, hints, label, source)
        walklist.append(data_point.to_dict())
    # data_point.dump_json(out_file)
    js = json.dumps(walklist, indent=4)
    
    with open(out_file, 'w') as op_file:
        op_file.write(js)
        print("Written the walks to " + out_file)


if __name__ == '__main__':
    main(sys.argv)
