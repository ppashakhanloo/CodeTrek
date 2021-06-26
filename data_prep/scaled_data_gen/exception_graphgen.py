import os
import sys
import argparse
import tempfile
import logging
import random
import multiprocessing
from joblib import Parallel, delayed


def run(tables_path):
  graph_bin = os.path.join(home_path, "data_prep/graph/build_graph.py")
  ex_stub_bin = os.path.join(home_path, "data_prep/random_walk/gen_stubs_exception.py")
  splits = tables_path.split("/")
  filename = splits[-1]

  task = splits[0]
  category = splits[1]
  label = splits[2]
  py_file = splits[3]

  try:
    with tempfile.TemporaryDirectory() as tables_dir:
      remote_tables_dir = os.path.join("gs://" + bucket_name, remote_table_dirname, tables_path)

      os.system("gsutil -m cp -r " + remote_tables_dir + "/*" + " " + tables_dir)
      os.system("python " + graph_bin + " " + tables_dir + " " + os.path.join(home_path, \
                "python_codeql/join.txt") + " " + tables_dir + "/graph_" + filename)
      assert os.path.exists(tables_dir + "/graph_" + filename + ".gv"), "graph not created."

      os.system("python " + ex_stub_bin + " " + tables_dir + "/graph_" + filename + ".gv" + " "\
        + tables_dir + " " + label + " " + tables_dir + "/stub_" + filename + ".json" + " " + walks_or_graphs)
      assert os.path.exists(tables_dir+"/stub_" + filename + ".json"), "stub not created."

      if os.path.exists(tables_dir + "/graph_" + filename + ".gv") and\
        os.path.exists(tables_dir + "/stub_" + filename + ".json"):
        os.system("gsutil cp  " + tables_dir + "/graph_" + filename + ".gv" + " " + "gs://" + bucket_name + \
                  "/" + output_graphs_dirname + "/" + task + "/" + category + "/" + "graph_" + filename + ".gv")
        os.system("gsutil cp  " + tables_dir + "/stub_" + filename + ".json" + " " + "gs://" + bucket_name + \
                  "/" + output_graphs_dirname + "/" + task + "/" + category + "/" + "stub_" + filename + ".json")

    with open(tables_paths_file + "-done", "a") as done:
      done.write(tables_path + "\n")
  except Exception as e:
    with open(tables_paths_file + "-log", "a") as log:
      log.write(">>" + tables_path + str(e) + "\n")

tables_paths_file = sys.argv[1] # paths.txt
bucket_name = sys.argv[2] # exception-storage
remote_table_dirname = sys.argv[3] # exception_tables
output_graphs_dirname = sys.argv[4] # output_large_graphs
home_path = sys.argv[5] # /home/pardisp/relational-representation
walks_or_graphs = sys.argv[6] # walks, graphs

programs = []

with open(tables_paths_file, 'r') as fin:
  for line in fin.readlines():  
    programs.append(line.strip())

Parallel(n_jobs=20, prefer="threads")(delayed(run)(program) for program in programs)

