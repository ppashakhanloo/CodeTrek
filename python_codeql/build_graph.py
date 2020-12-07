import csv
import os
import sys
import graphviz


def load_relation(facts_dir, fact_file):
    res = []
    with open(facts_dir + '/' + fact_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            res.append(tuple(line))

    return res

def load_joins(filepath):
    joins = {}
    keys = {}
    with open(filepath, 'r') as join_file:
        for line in join_file.readlines():
            tokens = line.split(' <----> ')
            assert len(tokens) == 2
            left = tokens[0].strip().split('.')
            left_rel = left[0].strip()
            left_index = int(left[1].strip())
            right = tokens[1].strip().split('.')
            right_rel = right[0].strip()
            right_index = int(right[1].strip())
            
            # joins
            if left_rel not in joins:
                joins[left_rel] = {}
            if right_rel not in joins[left_rel]:
                joins[left_rel][right_rel] = set()
            joins[left_rel][right_rel].add((left_index, right_index))
            if right_rel not in joins:
                joins[right_rel] = {}
            if left_rel not in joins[right_rel]:
                joins[right_rel][left_rel] = set()
            joins[right_rel][left_rel].add((right_index, left_index))

            # keys
            if left_rel not in keys:
                keys[left_rel] = set()
            keys[left_rel].add(left_index)
            if right_rel not in keys:
                keys[right_rel] = set()
            keys[right_rel].add(right_index)

    return joins, keys

def load_db(facts_dir):
    database = {}
    database["ip"] = {}
    for fact_file in os.listdir(facts_dir):
        if fact_file.endswith(".facts"):
            rulename = fact_file[:-10]
            database["ip"][rulename] = {
                "data": load_relation(facts_dir, fact_file)
            }

    return database

def build_value_to_tuple_map(db, keys):
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

def joinable(joins: dict, l_rel: str, l_tuple: tuple, r_rel: str, r_tuple: tuple) -> bool:
    if l_rel not in joins:
        return False
    if r_rel not in joins[l_rel]:
        return False
    pkfk_set = joins[l_rel][r_rel]
    for pkfk in pkfk_set:
        l_col = pkfk[0]
        r_col = pkfk[1]
        if l_tuple[l_col] == r_tuple[r_col]:
            return True
    return False

def edb_tuple_label(t: tuple, rel: str) -> str:
    return rel + '(' + ','.join(t) + ')'

def build_graph(db, joins, keys):
    edges = []
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
                    edges.append(tuple([index_i, index_j]))
                    graph.edge(str(index_i), str(index_j))

    return graph, edges

def save_edges(edges, output_file):
    with open(output_file, 'w') as f:
        for entry in edges:
            f.write(str(entry[0])+' '+str(entry[1])+'\n')

if __name__ == "__main__":
    facts_dir = sys.argv[1]
    join_filepath = sys.argv[2]
    output_file = sys.argv[3]

    db = load_db(facts_dir)
    joins, keys = load_joins(join_filepath)
    graph, edges = build_graph(db, joins, keys)
    # graph.render(output_file)
    save_edges(edges, output_file+'.edges')
