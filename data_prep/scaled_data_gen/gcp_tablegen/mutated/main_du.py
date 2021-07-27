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

class AllDoFn(beam.DoFn):
  
  def process(self, element):
    element = [s.strip() for s in element.split('/')]
    task = ''#element[-4]
    category = 'mutated_programs/defuse_rda/eval' #element[-3]
    label = element[-2]
    py_filename = element[-1]

    queries_dir = "codeql-runner/python_codeql/python-edb-queries/queries"

    try:
      if not os.path.isdir("codeql-runner"):
        os.system("gsutil -m cp gs://" + bucket_name + "/codeql-runner.tar.gz .")
        os.system("tar xfz codeql-runner.tar.gz")
        os.system("chmod -R 777 codeql-runner")
      assert os.path.isdir("codeql-runner"), "codeql-runner does not exist."

      remote_py = os.path.join("gs://" + bucket_name,
                                         task, category, label, py_filename)
      compute = False
      with tempfile.TemporaryDirectory() as src_dir:
        os.system("gsutil cp " + remote_py + " " + src_dir + "/" + py_filename)
        
        try:
          #s = os.system("gsutil cp " + "gs://" + bucket_name + "/tables/" + '/'.join(element) + "/" + "py_variables.bqrs.csv" + " " + src_dir + "/")
          #if s != 0:
          compute = True
          if not os.path.exists(src_dir + "/" + "py_variables.bqrs.csv"):
            compute = True
        except Exception as e:
          compute = True
          
        if compute:
          os.system("./codeql-runner/codeql_dir/codeql/codeql database create " + src_dir + "/db " + " --language=python --source-root=" + src_dir)

          assert os.path.isdir(src_dir+"/db"), "db does not exist."
 
          os.system("./codeql-runner/codeql_dir/codeql/codeql database run-queries " + src_dir + "/db " + queries_dir)
      
          bqrs_dir = src_dir + "/db/results/python-edb-queries/"
          for bqrs in os.listdir(bqrs_dir):
            if not bqrs.endswith("bqrs"):
              continue
            os.system("./codeql-runner/codeql_dir/codeql/codeql bqrs decode --entities=id\
             --output=" + bqrs_dir + "/" + bqrs + ".csv" + " --format=csv " + bqrs_dir + "/" + bqrs)
            assert bqrs_dir + "/" + bqrs + ".csv", "csv for " + bqrs + " does not exist."
        
          os.system("gsutil -m cp -r " + bqrs_dir + "/*.csv" + "\
            gs://" + bucket_name + "/" + output_tables + "/" + task + "/" + category + "/" + label + "/" + py_filename + "/")
          yield element
        else:
          yield "ALREADY FINISHED", str(element)
    except Exception as e:
      yield ">>", str(element), ">>", e
 
  
bucket_name = sys.argv[1] # varmisuse
region = sys.argv[2] # us-central1
py_files_name = sys.argv[3] # file.txt
output_tables = sys.argv[4] # tables


pipeline_options = PipelineOptions(flags=['--no_use_public_ips'], runner='DataflowRunner',\
  temp_location = 'gs://'+ bucket_name + '/temp', region=region, project='embeddings-4-static-analysis')
pipeline_options.view_as(SetupOptions).save_main_session = True
  
with beam.Pipeline(options=pipeline_options) as p:
  lines = p | "READ" >> ReadFromText("gs://" + bucket_name + "/" + py_files_name) | "Operate" >>\
  beam.ParDo(AllDoFn()) | "Write" >> WriteToText("gs://" + bucket_name + "/output/output")

