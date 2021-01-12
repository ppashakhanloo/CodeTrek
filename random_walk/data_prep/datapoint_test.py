import json
from data_prep.datapoint import DataPoint, Trajectory, TrajNode, TrajEdge


def main():
    v1 = TrajNode('v1')
    e1 = TrajNode('e1')
    s1 = TrajNode('s1')
    ev = TrajEdge('eid', 'vid')
    es = TrajEdge('eid', 'sid')
    traj1 = Trajectory(
        nodes=[e1, v1],
        edges=[ev]
    )
    traj2 = Trajectory(
        nodes=[e1, s1],
        edges=[es]
    )
    point = DataPoint(
        anchor=e1,
        trajectories=[traj1, traj2],
        hints=['pos', 'neg'],
        label='good'
    )
    json_str = json.dumps(point.to_dict(), indent=2)
    print(json_str)


if __name__ == '__main__':
    main()
