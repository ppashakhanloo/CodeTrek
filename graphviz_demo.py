import csv
import sys
import os
import graphviz


def load_relation(facts_dir: str, edb: str) -> list:
    res = []
    try:
        with open(facts_dir + '/' + edb, 'r') as edb_file:
            reader = csv.reader(edb_file, delimiter='\t')
            for line in reader:
                res.append(tuple(line))
        return res
    except:
        return []


def load_db(facts_dir: str, result_name: str) -> dict:
    database = {}
    database["op"] = {}
    database["ip"] = {}
    for edb_file in os.listdir(facts_dir):
        if edb_file.endswith(".facts"):
            rulename = edb_file[:-6]
            database["ip"][rulename] = {
                "data": load_relation(facts_dir, edb_file)
            }
    database["op"][result_name] = {
        "data_pos": load_relation(facts_dir, result_name + ".expected"),
        "data_neg": load_relation(facts_dir, result_name + ".undesired")
    }
    return database


# joins: relation name R1 -> relation name R2 -> (R1.c1, R2.c2)
# keys: relation name R -> a set of column indices that are keys in R
def load_joins(filepath: str) -> (dict, dict):
    joins = {}
    keys = {}
    with open(filepath, 'r') as join_file:
        for line in join_file.readlines():
            tokens = line.split(' <----> ')
            assert len(tokens) == 2
            left = tokens[0].strip().split('.')
            left_rel = left[0].strip()
            left_colinex = int(left[1].strip())
            right = tokens[1].strip().split('.')
            right_rel = right[0].strip()
            right_colindex = int(right[1].strip())
            # joins
            if left_rel not in joins:
                joins[left_rel] = {}
            if right_rel not in joins[left_rel]:
                joins[left_rel][right_rel] = set()
            joins[left_rel][right_rel].add((left_colinex, right_colindex))
            # keys
            if left_rel not in keys:
                keys[left_rel] = set()
            keys[left_rel].add(left_colinex)
            if right_rel not in keys:
                keys[right_rel] = set()
            keys[right_rel].add(right_colindex)
    return joins, keys


def build_value_to_tuple_map(db: dict, keys: dict) -> dict:
    val_tuple_map = {}
    for relname in db["ip"]:
        rel = db["ip"][relname]
        if relname in keys:
            indices = keys[relname]
            for entry in rel["data"]:
                for index in indices:
                    value = entry[index]
                    if value not in val_tuple_map:
                        val_tuple_map[value] = set()
                    val_tuple_map[value].add((entry, relname))
    return val_tuple_map


def edb_tuple_label(t: tuple, rel: str) -> str:
    return rel + '(' + ','.join(t) + ')'


def idb_tuple_label(t: tuple, rel: str) -> str:
    return rel + '((' + ','.join(t) + '))'


def joinable(joins: dict, l_rel: str, l_tuple: tuple, r_rel: str, r_tuple: tuple) -> bool:
    if l_rel not in joins:
        return False
    if r_rel not in joins[l_rel]:
        return False
    # NOTE: Add this to remove edges
    if r_rel == l_rel:
        return False
    pkfk_set = joins[l_rel][r_rel]
    for pkfk in pkfk_set:
        l_col = pkfk[0]
        r_col = pkfk[1]
        if l_tuple[l_col] == r_tuple[r_col]:
            return True
    return False


def build_graph(db: dict, joins: dict, keys: dict) -> graphviz.Graph:
    # EDB
    val_tuple_map = build_value_to_tuple_map(db, keys)
    tuple_index_map = {}
    index_rel_map = {}
    index_tuple_map = {}
    index = 1
    for relname in db["ip"]:
        rel = db["ip"][relname]
        for entry in rel["data"]:
            index_rel_map[index] = relname
            index_tuple_map[index] = entry
            tuple_index_map[entry] = index
            index += 1

    graph = graphviz.Graph()
    for i in index_tuple_map:
        graph.node(str(i), edb_tuple_label(index_tuple_map[i], index_rel_map[i]))
    for val in val_tuple_map:
        tuples = list(val_tuple_map[val])
        for i in range(len(tuples)):
            rel_i = tuples[i][1]
            tuple_i = tuples[i][0]
            index_i = tuple_index_map[tuple_i]
            for j in range(i+1, len(tuples)):
                rel_j = tuples[j][1]
                tuple_j = tuples[j][0]
                index_j = tuple_index_map[tuple_j]
                if joinable(joins, rel_i, tuple_i, rel_j, tuple_j):
                    graph.edge(str(index_i), str(index_j))
    # IDB
    for relname in db["op"]:
        rel = db["op"][relname]
        for entry in rel["data_pos"] + rel["data_neg"]:
            graph.node(str(index), idb_tuple_label(entry, relname), color='blue')
            for value in entry:
                # FIXME: Join based on type
                if value in val_tuple_map:
                    for t_r in val_tuple_map[value]:
                        index_t = tuple_index_map[t_r[0]]
                        graph.edge(str(index), str(index_t), color='blue')
            index += 1
    return graph


if __name__ == "__main__":
    if len(sys.argv) != 4:
         print("Usage: python3 graphviz_demo.py <facts_dir> <result_name> <join_file>")
         exit()
    facts_dir = sys.argv[1]
    result_name = sys.argv[2]
    join_filepath = sys.argv[3]
    # facts_dir = 'chown-whitelist_pruned'
    # result_name = 'cast_checker'
    # join_filepath = 'table_joins.txt'
    output_filepath = 'graph.gv'

    print("Loading database")
    db = load_db(facts_dir, result_name)
    print("Loading joins")
    joins, keys = load_joins(join_filepath)
    print("Building graph")
    graph = build_graph(db, joins, keys)
    print("Rendering graph")
    graph.render(output_filepath)
    print("Graph saved at", output_filepath)
