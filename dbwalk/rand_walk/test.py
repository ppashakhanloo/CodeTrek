import sys
import json
from typing import List
from dbwalk.rand_walk.walkutils import WalkUtils, JavaWalkUtils
from dbwalk.rand_walk.randomwalk import RandomWalker
from dbwalk.tokenizer import tokenizer


def main(args: List[str]) -> None:
    if not len(args) == 3:
        print('Usage: python3 test.py <graph-file> <json-file>')
        exit(1)

    gv_file = args[1]
    json_file = args[2]

    print('Loading graph')
    graph = RandomWalker.load_graph_from_gv(gv_file)

    print('Loading anchor and label')
    with open(json_file) as f:
        data = json.load(f)[0]
    anchor_str = data['anchor']
    label = data['label']  # not used in this test

    print('Generating random walks')
    language = 'python'
    num_walks = 3
    walker = RandomWalker(graph, anchor_str, language)
    # sample walks on the graph
    walks = walker.random_walk(max_num_walks=num_walks, min_num_steps=8, max_num_steps=16)
    # generate node and edge labels for each walk
    if language == 'python':
        trajectories = [WalkUtils.build_trajectory(walk) for walk in walks]
    elif language == 'java':
        trajectories = [JavaWalkUtils.build_trajectory(walk) for walk in walks]
    else:
        raise ValueError('Unknown language:', language)

    print(num_walks, 'walks generated')
    for trajectory in trajectories:
        print(trajectory.to_dict())

    print('Tokenizing nodes in trajectories:')
    LANG = 'python'
    for trajectory in trajectories:
        traj_dict = trajectory.to_dict()
        assert len(traj_dict['node_types']) == len(traj_dict['node_values'])
        for i in range(len(traj_dict['node_types'])):
            node_type = traj_dict['node_types'][i]
            node_value = traj_dict['node_values'][i]
            print('type tokens:', tokenizer.tokenize(node_type, LANG),
                  '. value tokens:', tokenizer.tokenize(node_value, LANG))


if __name__ == '__main__':
    main(sys.argv)
