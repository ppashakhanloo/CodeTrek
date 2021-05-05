import csv
import os
import graphviz
from io import StringIO
from data_prep.random_walk.walkutils import WalkUtils, JavaWalkUtils
from typing import List, Set, Dict, Tuple


class GraphBuilder:
    facts_dir = None      # type: str
    join_filepath = None  # type: str
    language = None       # type: str

    def __init__(self, facts_dir: str, join_filepath: str, language: str):
        self.facts_dir = facts_dir
        self.join_filepath = join_filepath
        self.language = GraphBuilder.parse_language(language)

    @staticmethod
    def parse_language(language: str) -> str:
        if language.lower() == 'python':
            return 'python'
        elif language.lower() == 'java':
            return 'java'
        else:
            raise ValueError('Unknown language:', language)

    def load_columns(self) -> Dict[str, List[str]]:
        if self.language == 'python':
            return WalkUtils.COLUMNS
        elif self.language == 'java':
            return JavaWalkUtils.COLUMNS
        else:
            raise ValueError('Cannot load columns for language:', self.language)

    def load_fact_table(self, fact_file: str) -> List[Tuple]:
        table = []
        with open(self.facts_dir + '/' + fact_file, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for line in reader:
                table.append(tuple(line))
        return table

    def load_csv_table(self, fact_file: str) -> List[Tuple]:
        table = []
        with open(self.facts_dir + '/' + fact_file, 'r') as f:
            data = f.read()
            data = data.replace('\x00', '**NULLBYTE**', -1)
            reader = csv.reader(StringIO(data), delimiter=',')
            rowlist = list(reader)
            for line in rowlist[1:]:
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

    def load_db(self, col_dict: Dict) -> Dict[str, List[Tuple]]:
        database = {}  # type: Dict[str, List[Tuple]]
        for fact_file in os.listdir(self.facts_dir):
            if fact_file.endswith('.csv.facts'):
                table_name = fact_file[:-10]  # remove ".csv.facts"
                if table_name in col_dict:
                    database[table_name] = self.load_fact_table(fact_file)
            elif fact_file.endswith('.bqrs.csv'):
                table_name = fact_file[:-9]  # remove ".bqrs.csv"
                if table_name in col_dict:
                    database[table_name] = self.load_csv_table(fact_file)
            elif fact_file.endswith('.csv'):
                table_name = fact_file[:-4]  # remove ".csv"
                if table_name in col_dict:
                    database[table_name] = self.load_csv_table(fact_file)
        return database

    def load_callgraph(self):
        if self.language == 'python':
            call_graph = []
            if 'one_hop_call_graph.csv.facts' in os.listdir(self.facts_dir):
                callgraph_file = 'one_hop_call_graph.csv.facts'
            elif 'full_call_graph.csv.facts' in os.listdir(self.facts_dir):
                callgraph_file = 'full_call_graph.csv.facts'
            elif 'one_hop_call_graph.csv' in os.listdir(self.facts_dir):
                callgraph_file = 'one_hop_call_graph.csv'
            elif 'full_call_graph.csv' in os.listdir(self.facts_dir):
                callgraph_file = 'full_call_graph.csv'
            elif 'one_hop_call_graph.bqrs.csv' in os.listdir(self.facts_dir):
                callgraph_file = 'one_hop_call_graph.bqrs.csv'
            elif 'full_call_graph.bqrs.csv' in os.listdir(self.facts_dir):
                callgraph_file = 'full_call_graph.bqrs.csv'
            else:
                return []

            with open(os.path.join(self.facts_dir, callgraph_file), 'r') as cf:
                for line in cf.readlines():
                    call_edge = line.split(',')
                    call_graph.append([s.strip() for s in call_edge])
            if callgraph_file.endswith('facts'):
                return call_graph
            else:
                return call_graph [1:]
        elif self.language == 'java':
            raise NotImplementedError
        else:
            raise ValueError('Cannot load the call graph for langauge:', self.language)

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
    def edb_tuple_label(t: tuple, rel: str) -> str:
        return rel + '(' + ','.join(t) + ')'

    @staticmethod
    def edb_edge_label(table1: str, col1: int, table2: str, col2: int) -> str:
        label1 = table1 + '.' + str(col1)
        label2 = table2 + '.' + str(col2)
        small = label1 if label1 <= label2 else label2
        large = label2 if label1 <= label2 else label1
        return '({l1},{l2})'.format(l1=small, l2=large)

    # returns a list of edge labels if two tuples are joinable
    # returns an empty list if two tuples are not joinable
    @staticmethod
    def labels_if_joinable(joins: Dict, columns: Dict, l_rel: str, l_tuple: Tuple, r_rel: str, r_tuple: Tuple) -> List[str]:
        if l_rel not in joins:
            return []
        if r_rel not in joins[l_rel]:
            return []
        labels = []
        pkfk_set = joins[l_rel][r_rel]
        for pkfk in pkfk_set:
            l_col_index = pkfk[0]
            r_col_index = pkfk[1]
            if l_tuple[l_col_index] == r_tuple[r_col_index]:
                l_col = columns[l_rel][l_col_index]
                r_col = columns[r_rel][r_col_index]
                labels.append(GraphBuilder.edb_edge_label(l_rel, l_col, r_rel, r_col))
        return labels

    def build_graph(self, db: Dict, joins: Dict, keys: Dict, columns: Dict, call_graph: List) -> graphviz.Graph:
        val_tuple_map = self.build_value_to_tuple_map(db, keys)  # type: Dict[str, Set[Tuple[Tuple, str]]]
        tuple_index_map = {}  # type: Dict[str, Dict[Tuple, int]]
        index_rel_map = {}    # type: Dict[int, str]
        index_tuple_map = {}  # type: Dict[int, Tuple]
        index = 1             # type: int
        # Build auxiliary maps
        for table_name in db:
            table = db[table_name]
            tuple_index_map[table_name] = {}
            for entry in table:
                index_rel_map[index] = table_name
                index_tuple_map[index] = entry
                tuple_index_map[table_name][entry] = index
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
                index_i = tuple_index_map[table_i][tuple_i]
                for j in range(i+1, len(tuples)):
                    table_j = tuples[j][1]
                    tuple_j = tuples[j][0]
                    index_j = tuple_index_map[table_j][tuple_j]
                    # generate all edges between two tuples
                    edge_labels = self.labels_if_joinable(joins, columns, table_i, tuple_i, table_j, tuple_j)
                    for edge_label in edge_labels:
                        graph.edge(str(index_i), str(index_j), edge_label)

        def search_py_exprs(expr_id):
            indices = []
            for i in index_rel_map:
                if index_rel_map[i] == 'py_exprs':
                    indices.append(i)
            for i in indices:
                if index_tuple_map[i][0] == expr_id:
                    return i
            return None

        for call_edge in call_graph:
            function1_expr_id = search_py_exprs(call_edge[0])
            function2_expr_id = search_py_exprs(call_edge[1])
            edge_label = '(callgraph.0,call_graph.1)'
            if function1_expr_id and function2_expr_id:
                graph.edge(str(function1_expr_id), str(function2_expr_id), edge_label)
        return graph

    def build(self, include_callgraph=False) -> graphviz.Graph:
        columns = self.load_columns()
        db = self.load_db(columns)
        if include_callgraph:
            callgraph = self.load_callgraph()
        else:
            callgraph = []
        joins, keys = self.load_joins()
        graph = self.build_graph(db, joins, keys, columns, callgraph)
        return graph

    @staticmethod
    def save_gv(graph: graphviz.Graph, output_file: str) -> None:
        with open(output_file, 'w') as f:
            f.write(str(graph))
