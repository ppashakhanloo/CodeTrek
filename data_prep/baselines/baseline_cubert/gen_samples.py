import os
import sys
import json
import tempfile

import networkx as nx
from joblib import Parallel, delayed

import code_to_subtoken
from data_prep.utils.utils import log
from data_prep.utils.gcp_utils import gcp_copy_from, gcp_copy_to
from data_prep.random_walk import gen_stubs_defuse as bc


def gen_exception(path):
  try:
    filename = path.split('/')[-1]
    prog_label = path.split('/')[-2]
    gcp_copy_from(path, tempfile.gettempdir(), bucket_name)
    with open(os.path.join(tempfile.gettempdir(), filename), 'r') as fin:
      code = fin.read()
    sample = {
      "function": code,
      "label": prog_label,
      "info": path
    }
    with open(os.path.join(tempfile.gettempdir(), filename), 'w') as f:
      json.dump(sample, f)
    gcp_copy_to(os.path.join(temfile.gettempdir(), filename),
                os.path.join(output_dirname, path.replace(prog_label+'/'+filename, '')),
                bucket_name)
    log(tables_paths_file + '-done', path)
  except Exception as e:
    log(tables_paths_file + '-error', path + str(e))

def gen_defuse(path, pred_kind):
  try:
    filename = path.split('/')[-1]
    prog_label = path.split('/')[-2]
    with tempfile.TemporaryDirectory() as tables_dir:
      gcp_copy_from(path, tempfile.gettempdir(), bucket_name)
      if pred_kind == 'prog_cls':
        with open(os.path.join(tempfile.gettempdir(), filename), 'r') as fin:
          code = fin.read()
        sample = {
          "function": code,
          "label": prog_label,
          "info": path
        }
        with open(os.path.join(tempfile.gettempdir(), 'sample_' + filename + '.json'), 'w') as f:
          json.dump(sample, f)
        gcp_copy_to(os.path.join(tempfile.gettempdir(), 'sample_' + filename + '.json'),
                    os.path.join(output_dirname, path.replace(prog_label+'/'+filename, '')),
                    bucket_name)
        return
      if pred_kind == 'loc_cls':
        gcp_copy_from(os.path.join(remote_table_dirname, path, 'unused_var.bqrs.csv'),
                      tables_dir, bucket_name)
        gcp_copy_from(os.path.join(remote_table_dirname, path, 'py_expr_contexts.bqrs.csv'),
                      tables_dir, bucket_name)
        gcp_copy_from(os.path.join(remote_table_dirname, path, 'py_exprs.bqrs.csv'),
                      tables_dir, bucket_name)
        gcp_copy_from(os.path.join(remote_table_dirname, path, 'locations_ast.bqrs.csv'),
                      tables_dir, bucket_name)
        gcp_copy_from(os.path.join(remote_table_dirname, path, 'py_locations.bqrs.csv'),
                      tables_dir, bucket_name)
        subtokens, code = [], []
        with open(tempfile.gettempdir()+'/'+filename, 'r') as f:
          lines = f.readlines()
          for line in lines:
            code.append(line)
            st = code_to_subtoken.subtokenize_string(vocab, line)
            subtokens.append(st[0] if len(st) > 0 else st)
        all_defs = bc.get_semmle_defs(tables_dir)
        unused_vars = bc.get_semmle_unused_vars(tables_dir)
        ast_locs, py_locs = bc.get_semmle_locs(tables_dir)

        for d in all_defs:
          # first, find whether it's used on not
          label = "used"
          for u in unused_vars:
            if u[0] == d[0]:
              label = "unused"
              break
          # now, find the location
          def_loc = None
          for py_loc in py_locs:
            if py_loc[1] == d[0]:
              def_loc = py_loc
              break
          if def_loc:
            for ast_loc in ast_locs:
              if def_loc[0] == ast_loc[0]:
                def_loc = ast_loc
                break
          flat_subtokens = []
          for s in subtokens:
            flat_subtokens += s
          mask = [0] * len(flat_subtokens)

          row0, col0 = int(def_loc[2]), int(def_loc[3])
          row1, col1 = int(def_loc[4]), int(def_loc[5])
          var_name = code[row0 - 1][col0 - 1:col1]
          st = (code_to_subtoken.subtokenize_string(vocab, var_name))[0][0]
          ind = subtokens[row0 - 1].index(st)
          # create the mask:
          num_before, num_after = 0, 0
          for i in range(0, row0 - 1):
            num_before += len(subtokens[i])
          num_before += ind
          for i in range(row0 - 1, len(subtokens)):
            num_after += len(subtokens[i])
          num_after -= ind
          mask[num_before] = 1
          sample = {
            "function": flat_subtokens,
            "label": label,
            "location": mask,
            "info": path
          }
          with open(os.path.join(tables_dir, 'sample_' + d[0] + '_' + filename + '.json'), 'w') as f:
            json.dump(sample, f)
          gcp_copy_to(os.path.join(tables_dir, 'sample_' + d[0] + '_' + filename + '.json'),
                      os.path.join(output_dirname, path.replace(prog_label + '/' + filename, '')),
                      bucket_name)
    log(tables_paths_file + '-done', path)
  except Exception as e:
    raise e
    log(tables_paths_file + '-error', path + str(e))

def gen_varmisuse(path, pred_kind):
  try:
    filename = path.split('/')[-1]
    prog_label = path.split('/')[-2]
    with tempfile.TemporaryDirectory() as tables_dir:
      path_prefix = path.replace(prog_label + '/' + filename, '')[:-1]
      num = int(filename[5:-3])
      if prog_label == "correct":
        file_1_src = "file_" + str(num) + ".py"
        file_1 = path_prefix + "/correct/" + file_1_src
        file_2_src = "file_" + str(num + 1) + ".py"
        file_2 = path_prefix + "/misuse/" + file_2_src
      else:
        file_1_src = "file_" + str(num) + ".py"
        file_1 = path_prefix + "/misuse/" + file_1_src
        file_2_src = "file_" + str(num - 1) + ".py"
        file_2 = path_prefix + "/correct/" + file_2_src
      gcp_copy_from(file_1, os.path.join(tables_dir, file_1_src), bucket_name)
      gcp_copy_from(file_2, os.path.join(tables_dir, file_2_src), bucket_name)
      if pred_kind == 'loc_cls':
        raise NotImplementedError(pred_kind)
      with open(tables_dir + '/sample_' + filename + '.json', 'w') as f:
        json.dump(point, f)
      gcp_copy_to(os.path.join(tables_dir, 'sample_'+filename+'.json'),
                  os.path.join(output_dirname, path.replace(prog_label+'/'+filename)),
                  bucket_name)
    log(tables_paths_file + '-done', path)
  except Exception as e:
    log(tables_paths_file + '-error', path + str(e))

if __name__ == "__main__":
  tables_paths_file = sys.argv[1] # paths.txt
  bucket_name = sys.argv[2] # generated-tables
  remote_table_dirname = sys.argv[3] # outdir_reshuffle
  output_dirname = sys.argv[4] # output_samples
  vocab = sys.argv[5] # vocab.txt
  task_name = sys.argv[6]
  assert task_name in ['defuse', 'exception', 'varmisuse']
  pred_kind = sys.argv[7]
  assert pred_kind in ['prog_cls', 'loc_cls', 'loc_rep']

  with open(tables_paths_file, 'r') as fin:
    paths = [line.strip() for line in fin.readlines()]

  if task_name == 'varmisuse':
    Parallel(n_jobs=10, prefer="threads")(delayed(gen_varmisuse)(path, pred_kind) for path in paths)
  if task_name == 'defuse':
    Parallel(n_jobs=10, prefer="threads")(delayed(gen_defuse)(path, pred_kind) for path in paths)
  if task_name == 'exception':
    Parallel(n_jobs=10, prefer="threads")(delayed(gen_exception)(path) for path in paths)
