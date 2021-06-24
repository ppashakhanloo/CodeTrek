import apache_beam as beam
import argparse
import os
import tempfile
import logging
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
import sys
import json
import pickle as cp

class ExtractDictDoFn(beam.DoFn):
  def setup(self):
    os.system("pip install javalang")
    os.system("gsutil cp gs://" + bucket_name + "/" + labels_file + " " + ".")
    os.system("gsutil -m cp -r gs://exception-small/data_prep .")

  def load_label_dict(self, labels_file):
    label_dict = {}
    assert os.path.exists(labels_file), 'labels_file does not exist.'
    with open(labels_file, 'r') as f:
      for i, row in enumerate(f):
        label = row.strip()
        assert not label in label_dict, 'duplicated labels'
        label_dict[label] = i
    return label_dict

  def load_graph_online(self, fname):
    nodes = []
    edges = []
    with open(fname, 'r') as f:
      lines = f.readlines()
      lines = lines[1:-1]
      for line in lines:
        if '--' in line and '[label=' in line:
          splits = line.strip().split(' ')
          label = splits[3].strip()[1:-1].replace('label="', '')[:-1] 
          edges.append((splits[0].strip(),splits[2].strip(),label))
        elif '[label=' in line:
          splits = line.strip().split(' ')
          label = splits[1].strip()[1:-1].replace('label="', '')[:-1]
          nodes.append((splits[0].strip(), label))
    return nodes, edges

  def process(self, element):
    label_dict = self.load_label_dict(labels_file)
    element = [s.strip() for s in element.split(',')]
    file_path = element[0]
    fname = file_path.split('/')[-1]

    node_types = set()
    edge_types = set()
    token_vocab = set()

    for key in ['__pad__', '__unknown__']:
      node_types.add(key)
      edge_types.add(key)
      token_vocab.add(key)

    try:
      assert os.path.exists('data_prep'), 'data_prep does not exist.'
      os.system("gsutil cp gs://" + bucket_name + "/" + file_path + " " + ".")
      sys.path.append(os.getcwd())
      from data_prep.tokenizer import tokenizer
      from data_prep.random_walk.walkutils import WalkUtils
      assert os.path.exists(fname), 'fname does not exist.'
      graph_nodes, graph_edges = self.load_graph_online(fname)
      var_set = set()
      for node in graph_nodes:
        node_idx = node[0]
        node_label = node[1]
        node_name, values = WalkUtils.parse_node_label(node_label)
        node_type, node_value = WalkUtils.gen_node_type_value(node_name, values)

        if node_type.startswith('v_'):
          var_set.add(node_type)
        else:
          node_types.add(node_type)
        if len(node_value) != 0:
          tok = tokenizer.tokenize(node_value, 'python')
        else:
          tok = []
        for key in tok:
          token_vocab.add(key)

      for e in graph_edges:
        edge = e[2]
        edge_types.add(edge)

      yield node_types, edge_types, token_vocab
    except Exception as e:
      yield node_types, edge_types, token_vocab

def combine(elem):
  node_types_agg = set()
  edge_types_agg = set()
  token_vocab_agg = set()
  for e in elem:
    node_types_agg.update(e[0])
    edge_types_agg.update(e[1])
    token_vocab_agg.update(e[2])
  return node_types_agg, edge_types_agg, token_vocab_agg

def get_or_add(type_dict, key):
  if key in type_dict:
    return type_dict[key]
  val = len(type_dict)
  type_dict[key] = val
  return val

def var_idx2name(idx):
  return 'var_%d' % idx

def dump_dict(elem):
  node_types = elem[0]
  edge_types = elem[1]
  token_vocab = elem[2]

  node_types_fin = {}
  edge_types_fin = {}
  token_vocab_fin = {}
  for i in node_types:
    get_or_add(node_types_fin, i)
  for i in edge_types:
    get_or_add(edge_types_fin, i)
  for i in token_vocab:
    get_or_add(token_vocab_fin, i)

  var_dict = {}
  var_reverse_dict = {}
  for i in range(max_num_vars):
    val = get_or_add(node_types_fin, var_idx2name(i))
    var_dict[i] = val
    var_reverse_dict[val] = i

  with open('dict.pkl', 'wb') as f:
    d = {}
    d['node_types'] = node_types_fin
    d['edge_types'] = edge_types_fin
    d['n_vars'] = max_num_vars
    d['var_dict'] = var_dict
    d['token_vocab'] = token_vocab_fin
    d['var_reverse_dict'] = var_reverse_dict
    cp.dump(d, f, cp.DEFAULT_PROTOCOL)

  os.system("gsutil cp " + "dict.pkl" + " " + "gs://" + bucket_name + "/")

bucket_name = sys.argv[1] # generated-tables
region = sys.argv[2] # us-central1
names_list = sys.argv[3] # paths.txt
labels_file = sys.argv[4] # vm_labels.txt
max_num_vars = 100

home_path=os.getcwd()
pipeline_options = PipelineOptions(flags=['--no_use_public_ips'], runner='DataflowRunner',\
  temp_location = 'gs://'+ bucket_name + '/temp', region=region, project='embeddings-4-static-analysis')
pipeline_options.view_as(SetupOptions).save_main_session = True

with beam.Pipeline(options=pipeline_options) as p:
  p | "READ" >> ReadFromText("gs://" + bucket_name + "/" + names_list) | "Operate" >>\
  beam.ParDo(ExtractDictDoFn()) | beam.CombineGlobally(combine) | beam.FlatMap(dump_dict)
