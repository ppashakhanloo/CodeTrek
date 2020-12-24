import csv
import os
import sys
import graphviz
from typing import List, Set, Dict, Tuple


class GraphBuilder:
    facts_dir = None      # type: str
    join_filepath = None  # type: str

    def __init__(self, facts_dir: str, join_filepath: str):
        self.facts_dir = facts_dir
        self.join_filepath = join_filepath

    def load_table(self, fact_file: str) -> List[Tuple]:
        table = []
        with open(self.facts_dir + '/' + fact_file, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for line in reader:
                table.append(tuple(line))
        return table

    def load_joins(self) -> (Dict, Dict):
        joins = {}  # type: Dict[str, Dict[str, Set[(int, int)]]]
        keys = {}   # type: Dict[str, Set[int]]
        with open(self.join_filepath, 'r') as join_file:
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

    def load_db(self) -> Dict[str, List[Tuple]]:
        database = {}  # type: Dict[str, List[Tuple]]
        for fact_file in os.listdir(self.facts_dir):
            if fact_file.endswith('.csv.facts'):
                table_name = fact_file[:-10]  # remove ".csv.facts"
                database[table_name] = self.load_table(fact_file)
        return database

    @staticmethod
    def build_value_to_tuple_map(db: Dict, keys: Dict) -> Dict[str, Set[Tuple[Tuple, str]]]:
        val_tuple_map = {}  # type: Dict[str, Set[Tuple[Tuple, str]]]
        for table_name in db:
            table = db[table_name]
            if table_name in keys:
                indices = keys[table_name]
                for entry in table:
                    for index in indices:
                        value = entry[index]
                        if value not in val_tuple_map:
                            val_tuple_map[value] = set()
                        val_tuple_map[value].add((entry, table_name))
        return val_tuple_map

    @staticmethod
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

    @staticmethod
    def edb_tuple_label(t: tuple, rel: str) -> str:
        return rel + '(' + ','.join(t) + ')'

    def build_graph(self, db: Dict, joins: Dict, keys: Dict) -> graphviz.Graph:
        val_tuple_map = self.build_value_to_tuple_map(db, keys)  # type: Dict[str, Set[(Tuple, str)]]
        tuple_index_map = {}  # type: Dict[Tuple, int]
        index_rel_map = {}    # type: Dict[int, str]
        index_tuple_map = {}  # type: Dict[int, Tuple]
        index = 1             # type: int
        # Build auxiliary maps
        for table_name in db:
            table = db[table_name]
            for entry in table:
                index_rel_map[index] = table_name
                index_tuple_map[index] = entry
                tuple_index_map[entry] = index
                index += 1
        # Build the graph
        graph = graphviz.Graph()
        for i in index_tuple_map:
            graph.node(str(i), self.edb_tuple_label(index_tuple_map[i], index_rel_map[i]))
        for val in val_tuple_map:
            tuples = list(val_tuple_map[val])
            for i in range(len(tuples)):
                table_i = tuples[i][1]
                tuple_i = tuples[i][0]
                index_i = tuple_index_map[tuple_i]
                for j in range(i+1, len(tuples)):
                    table_j = tuples[j][1]
                    tuple_j = tuples[j][0]
                    index_j = tuple_index_map[tuple_j]
                    if self.joinable(joins, table_i, tuple_i, table_j, tuple_j):
                        graph.edge(str(index_i), str(index_j))
        return graph

    def build(self) -> graphviz.Graph:
        db = self.load_db()
        joins, keys = self.load_joins()
        graph = self.build_graph(db, joins, keys)
        return graph

    @staticmethod
    def save_gv(graph: graphviz.Graph, output_file: str) -> None:
        with open(output_file, 'w') as f:
            f.write(str(graph))


def main(args: List[str]) -> None:
    if not len(args) == 4:
        print('Usage: python3 build_graph.py <facts_dir> <join_filepath> <output_file>')
        exit(1)

    facts_dir = args[1]
    join_filepath = args[2]
    output_file = args[3]

    graph_builder = GraphBuilder(facts_dir, join_filepath)
    graph = graph_builder.build()
    GraphBuilder.save_gv(graph, output_file + '.gv')


if __name__ == "__main__":
    main(sys.argv)
