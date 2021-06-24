import sys
import csv
import json
import os
from pygraphviz import Node
from data_prep.random_walk.datapoint import DataPoint, TrajNodeValue
from data_prep.random_walk.walkutils import WalkUtils
from data_prep.random_walk.randomwalk import RandomWalker
from typing import List

MAX_NUM_WALKS=100

def get_anchor(edb_dir: str, label: str, pred_kind: str):
    diff_len = []
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
                    diff_len.append(len(row[1]) - len(row[7]))
            else:
                assert label == "misuse"
                loc = [ x.strip() for x in row[6:-1] ] + [str(int(row[-1])-1)]
                if loc[1][0] == "@":
                    loc[3] = str(int(loc[3]) + 1)
                if loc not in misuse_locs:
                    misuse_locs.append(loc)
                    diff_len.append(len(row[1]) - len(row[7]))

    assert len(misuse_locs) > 0
    m_loc = misuse_locs[0]
    if diff_len[0] != 0:
        misuse_locs.append([m_loc[0], m_loc[1], m_loc[2], m_loc[3], m_loc[4], str(int(m_loc[5])+diff_len[0])])
 
    misuse_loc_ids = []
    with open(os.path.join(edb_dir, "locations_ast.bqrs.csv")) as locations_ast,\
            open(os.path.join(edb_dir, "py_locations.bqrs.csv")) as py_locations:
        ast_loc_reader = csv.reader(locations_ast, delimiter=',')
        ast_loc_rows = list(ast_loc_reader)
        ast_loc_ids = []
        for loc in misuse_locs:
            for row in ast_loc_rows:
                if loc[2:] == row[2:]:
                    ast_loc_ids.append(row)
        py_loc_reader = csv.reader(py_locations, delimiter=',')
        py_loc_rows = list(py_loc_reader)
        for loc in ast_loc_ids:
            for row in py_loc_rows:
                if loc[0] == row[0]:
                    misuse_loc_ids.append(row)
    assert len(misuse_loc_ids) >= 1

    anchors = set()
    with open(os.path.join(edb_dir, "py_variables.bqrs.csv")) as py_variables:
        reader = csv.reader(py_variables, delimiter=',')
        for iter, row in enumerate(reader):
            if pred_kind == 'prog_cls':
                if iter == 0:
                    continue
                anchors.add('py_variables(' + ','.join(row) + ')')
            elif pred_kind == 'loc_cls':
                for id in misuse_loc_ids:
                    if row[1] == id[1]:
                        anchors.add('py_variables(' + ','.join(row) + ')')
            else:
                raise NotImplementedError(pred_kind)
    assert len(anchors) >= 1
    return anchors


def main(args: List[str]) -> None:
    gv_file = args[1]
    edb_path = args[2]
    gt = args[3]
    assert gt in ["misuse", "correct"]
    out_file = args[4]
    walks_or_graphs = args[5]
    assert walks_or_graphs in ['walks', 'graphs']
    pred_kind = args[6]
    assert pred_kind in ['prog_cls', 'loc_cls']

    anchor_nodes = set()
    anchors = get_anchor(edb_path, gt, pred_kind)
    graph = RandomWalker.load_graph_from_gv(gv_file)

    walklist = []
    for node in graph.nodes():
        element = graph.nodes[node]['label']
        if element in anchors:
            anchor_nodes.add((node, gt))

    traj_anchors = []
    walks_all = []
    for anchor, _ in anchor_nodes:
        anchor_label = graph.nodes[anchor]['label']
        if walks_or_graphs == 'graphs':
            walks = []
        else:
            random_walk = RandomWalker(graph, anchor_label, 'python')
            walks = \
                random_walk.random_walk(max_num_walks=MAX_NUM_WALKS//len(anchor_nodes), \
                                        min_num_steps=8, max_num_steps=24)
        traj_anchors.append(TrajNodeValue(anchor_label))
        walks_all += walks
    if walks_or_graphs == 'walks':
        trajectories = [WalkUtils.build_trajectory(walk) for walk in RandomWalker.padding(walks, MAX_NUM_WALKS)]
    else:
        trajectories = []
    data_point = DataPoint(traj_anchors, trajectories, [], gt, gv_file)
    walklist.append(data_point.to_dict())
    js = json.dumps(walklist)

    with open(out_file, 'w') as op_file:
        op_file.write(js)

if __name__ == '__main__':
    main(sys.argv)
