import os
import sys
import json
from os.path import join
from shutil import copyfile
from pathlib import Path
from typing import List


def gen_graph_file(src: str, tgt: str) -> None:
    copyfile(src, tgt)


def gen_json_file(src: str, tgt: str) -> None:
    with open(src) as f:
        data = json.load(f)
    assert len(data) == 1
    data[0]['trajectories'] = []
    data[0]['hints'] = []
    with open(tgt, 'w') as outfile:
        outfile.write(json.dumps(data, indent=4))


def collect_files(walks_dir: str, out_dir: str) -> None:
    # create new directories
    Path(join(out_dir, 'train')).mkdir(parents=True, exist_ok=True)
    Path(join(out_dir, 'dev')).mkdir(parents=True, exist_ok=True)
    Path(join(out_dir, 'eval')).mkdir(parents=True, exist_ok=True)

    for name in os.listdir(walks_dir):
        # compute graph and json file names
        tokens = name.split('_')
        assert len(tokens) == 5
        dir_name = tokens[2]
        gv_name = tokens[3] + '_' + tokens[4] + '_graph.gv'
        graph_file = join(join(walks_dir, name), gv_name)
        json_file = join(join(walks_dir, name), 'walks.json')

        # generate the new files
        new_json_name = 'walks_exception_' + dir_name + '_file_' + tokens[4] + '.json'
        gen_json_file(json_file, join(join(out_dir, dir_name), new_json_name))
        new_graph_name = 'graph_exception_' + dir_name + '_file_' + tokens[4] + '.gv'
        gen_graph_file(graph_file, join(join(out_dir, dir_name), new_graph_name))


def main(args: List[str]) -> None:
    if not len(args) == 3:
        print('Usage: python3 collect_files.py <walks-dir> <out-dir>')
    walks_dir = args[1]
    out_dir = args[2]
    collect_files(walks_dir, out_dir)


if __name__ == '__main__':
    main(sys.argv)
