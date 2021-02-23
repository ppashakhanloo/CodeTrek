import os
import sys
import json
from graphviz import Graph
from pathlib import Path
from shutil import copyfile
from typing import Dict
from dbwalk.graph.graphutils import GraphUtils


class LabelSimplifier:
    EDGE_LABEL_PREFIX = 'e,'  # type: str
    gv_dir_path = None        # type: str
    out_dir_path = None       # type: str
    edge_label_map = {}       # type: Dict[str, str]
    edge_label_index = 0      # type: int
    output_type = None        # type: str

    def __init__(self, gv_dir: str, out_dir: str, output_type: str):
        self.gv_dir_path = gv_dir
        self.out_dir_path = out_dir
        self.output_type = output_type.lower()  # either 'bin' or 'dot'

    def convert(self, in_path: str, out_path: str):
        new_graph = Graph()
        graph = GraphUtils.load_graph_from_gv(in_path)
        for node in graph.nodes():
            label = graph.nodes()[node]['label']
            new_graph.node(node, label)
        for n1, n2 in graph.edges():
            labels = {edge['label'] for edge in list(graph[n1][n2].values())}
            for label in labels:
                if label not in self.edge_label_map.keys():
                    self.edge_label_index += 1
                    self.edge_label_map[label] = self.EDGE_LABEL_PREFIX + str(self.edge_label_index)
                new_graph.edge(n1, n2, self.edge_label_map[label])
        if self.output_type == 'bin':
            GraphUtils.serialize(new_graph, out_path)
        elif self.output_type == 'dot':
            GraphUtils.save_gv(new_graph, out_path)
        else:
            raise ValueError(self.output_type)

    def convert_all(self):
        for in_dir, _, files in os.walk(gv_dir_path):
            out_dir = self.out_dir_path + in_dir[len(gv_dir_path):]
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            for name in files:
                in_path = os.path.join(in_dir, name)
                out_path = os.path.join(out_dir, name)
                if name.endswith('.gv'):
                    if self.output_type == 'bin':
                        out_path += 'b'
                    print('Convert:', in_path, '-->', out_path)
                    self.convert(in_path, out_path)
                else:
                    print('Copy:', in_path, '-->', out_path)
                    copyfile(in_path, out_path)

    def dump_map(self, filepath: str):
        with open(filepath, 'w') as outfile:
            json.dump(self.edge_label_map, outfile, indent=4)


if __name__ == '__main__':
    args = sys.argv
    if not len(args) == 4:
        print('Usage: python3 simplify_labels.py <gv-dir-path> <out-dir> <out-dict-path>')
        exit(1)

    gv_dir_path = args[1]
    out_dir_path = args[2]
    out_dict_path = args[3]

    simplifier = LabelSimplifier(gv_dir_path, out_dir_path, output_type='bin')
    simplifier.convert_all()
    simplifier.dump_map(out_dict_path)
