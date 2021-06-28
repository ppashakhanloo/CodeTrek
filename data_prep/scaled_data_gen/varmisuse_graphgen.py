import argparse
import os
import tempfile
import logging
import random
from joblib import Parallel, delayed
import sys


def run(tables_path):
  graph_bin = os.path.join(home_path, "data_prep/graph/build_graph.py")
  varmisuse_stub_bin = os.path.join(home_path, "data_prep/random_walk/gen_stubs_varmisuse.py")
  splits = tables_path.split("/")
  filename = splits[-1]

  task = splits[0]
  category = splits[1]
  label = splits[2]
  py_file = splits[3]

  num = int(py_file[5:-3])

  if label == "correct":
    file_1 = "gs://" + bucket_name + "/varmisuse/" + category + "/" + "correct" + "/" + "file_" + str(num) + ".py"
    file_1_src = "file_" + str(num) + ".py"
    file_2 = "gs://" + bucket_name + "/varmisuse/" + category + "/" + "misuse" + "/" + "file_" + str(num + 1) + ".py"
    file_2_src = "file_" + str(num + 1) + ".py"
  else:
    file_1 = "gs://" + bucket_name + "/varmisuse/" + category + "/" + "misuse" + "/" + "file_" + str(num) + ".py"
    file_1_src = "file_" + str(num) + ".py"
    file_2 = "gs://" + bucket_name + "/varmisuse/" + category + "/" + "correct" + "/" + "file_" + str(num - 1) + ".py"
    file_2_src = "file_" + str(num - 1) + ".py"

  try:
    with tempfile.TemporaryDirectory() as tables_dir:
      os.system("gsutil cp  " + file_1 + " " + tables_dir + "/" + file_1_src)
      os.system("gsutil cp  " + file_2 + " " + tables_dir + "/" + file_2_src)

      remote_tables_dir = "gs://" + bucket_name + "/" + remote_table_dirname + "/" + tables_path
      diff_bin = os.path.join(home_path, 'data_prep/random_walk/diff.py')

      os.system("gsutil -m cp  -r " + remote_tables_dir + "/*" + " " + tables_dir)
      os.system("python " + diff_bin + " " + tables_dir + "/" + file_1_src + " " + tables_dir + "/" + file_2_src + " " + tables_dir + "/var_misuses.csv")
      os.system("python " + graph_bin + " " + tables_dir + " " + os.path.join(home_path, "python_codeql/join.txt") + " " + tables_dir + "/graph_" + filename)
      assert os.path.exists(tables_dir + "/graph_" + filename + ".gv"), "graph not created."

      os.system("python " + varmisuse_stub_bin + " " + tables_dir + "/graph_" + filename + ".gv" + " "\
        + tables_dir + " " + label + " " + tables_dir + "/stub_" + filename + ".json" + " " + walks_or_graphs + " " + pred_kind)
      assert os.path.exists(tables_dir + "/stub_" + filename + ".json"), "stub vm not created."

      if os.path.exists(tables_dir + "/graph_" + filename + ".gv") and os.path.exists(tables_dir + "/stub_" + filename + ".json"):
        os.system("gsutil cp  " + tables_dir + "/graph_" + filename + ".gv" + " " + \
                  "gs://" + bucket_name + "/" + output_graphs_dirname + "/" + \
                  task + "/" + category + "/" + "graph_" + filename + ".gv")
        os.system("gsutil cp  " + tables_dir + "/stub_" + filename + ".json" + " " + \
                  "gs://" + bucket_name + "/" + output_graphs_dirname + "/" + \
                  task + "/" + category + "/" + "stub_" + filename + ".json")

    with open(tables_paths_file + "-done", "a") as done:
      done.write(tables_path + "\n")
  except Exception as e:
    with open(tables_paths_file + "-log", "a") as log:
      log.write(">>" + tables_path + str(e) + "\n")


tables_paths_file = sys.argv[1] # paths.txt
bucket_name = sys.argv[2] # generated-tables
remote_table_dirname = sys.argv[3] # outdir_reshuffle
output_graphs_dirname = sys.argv[4] # output_graphs
home_path = sys.argv[5] # /home/pardisp/relational-representation
walks_or_graphs = sys.argv[6] # walks, graphs
pred_kind = sys.argv[7] # prog_cls, loc_cls, loc_rep

programs = []

import multiprocessing
with open(tables_paths_file, 'r') as fin:
  for line in fin.readlines():  
    programs.append(line.strip())

Parallel(n_jobs=10, prefer="threads")(delayed(run)(program) for program in programs)
