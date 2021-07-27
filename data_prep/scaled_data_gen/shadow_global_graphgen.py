import argparse
import os
import tempfile
import logging
import random
from joblib import Parallel, delayed
import sys


def run(tables_path):
  graph_bin = os.path.join(home_path, "data_prep/graph/build_graph.py")
  shadow_global_stub_bin = os.path.join(home_path, "data_prep/random_walk/gen_stubs_shadow_global.py")
  splits = tables_path.split("/")
  filename = splits[-1]

  task = 'shadowglobal'
  category = splits[-2]
  py_file = splits[-1]

  filepath = "gs://" + bucket_name + "/logic-tasks/" + category + "/" + py_file

  try:
    with tempfile.TemporaryDirectory() as tables_dir:
      os.system("gsutil cp  " + filepath + " " + tables_dir + "/" + py_file)

      remote_tables_dir = "gs://" + bucket_name + "/" + remote_table_dirname + "/" + tables_path

      os.system("gsutil -m cp  -r " + remote_tables_dir + "/*" + " " + tables_dir)
      os.system("python " + graph_bin + " " + tables_dir + " " + os.path.join(home_path, "python_codeql/join.txt") + " " + tables_dir + "/graph_" + filename)
      assert os.path.exists(tables_dir + "/graph_" + filename + ".gv"), "graph not created."

      os.system("python " + shadow_global_stub_bin + " " + tables_dir + "/graph_" + filename + ".gv" + " "\
        + tables_dir + " " + tables_dir + "/stub_" + filename + ".json" + " " + walks_or_graphs + " " + pred_kind)
      assert os.path.exists(tables_dir + "/stub_" + filename + ".json"), "stub not created."

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
bucket_name = 'logic-tasks'  #sys.argv[2] # generated-tables
remote_table_dirname = 'tables'  #sys.argv[3] # outdir_reshuffle
output_graphs_dirname = 'walks_shadow_global'  #sys.argv[4] # output_graphs
home_path = '/home/pardisp/relational-representation'
walks_or_graphs = sys.argv[2] # walks, graphs
pred_kind = 'prog_cls'

programs = []

import multiprocessing
with open(tables_paths_file, 'r') as fin:
  for line in fin.readlines():  
    programs.append(line.strip())

Parallel(n_jobs=25, prefer="threads")(delayed(run)(program) for program in programs)
