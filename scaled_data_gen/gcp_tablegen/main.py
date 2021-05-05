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
    task = element[0]
    category = element[1]
    label = element[2]
    py_filename = element[3]

    queries_dir = "codeql-runner/python_codeql/python-edb-queries/task_" + task + "_queries"
    
    try:
      if not os.path.isdir("codeql-runner"):
        os.system("gsutil -m cp gs://generated-tables/codeql-runner.tar.gz .")
        os.system("tar xfz codeql-runner.tar.gz")
        os.system("chmod -R 777 codeql-runner")
      assert os.path.isdir("codeql-runner"), "codeql-runner does not exist."

      remote_py = os.path.join("gs://" + bucket_name,
                                         task, category, label, py_filename)
      
      with tempfile.TemporaryDirectory() as src_dir:
        os.system("gsutil cp " + remote_py + " " + src_dir + "/" + py_filename)
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
    except Exception as e:
      yield ">>", element, ">>", e
 
  
bucket_name = sys.argv[1] # generated-tables
region = sys.argv[2] # us-central1
py_files_name = sys.argv[3] # file.txt
output_tables = sys.argv[4] # outdir_reshuffle


pipeline_options = PipelineOptions(flags=['--no_use_public_ips'], runner='DirectRunner',\
  temp_location = 'gs://'+ bucket_name + '/temp', region=region, project='embeddings-4-static-analysis')
pipeline_options.view_as(SetupOptions).save_main_session = True
  
with beam.Pipeline(options=pipeline_options) as p:
  lines = p | "READ" >> ReadFromText("gs://" + bucket_name + "/" + py_files_name) | "Reshuffle_1" >> beam.Reshuffle() | "Operate" >>\
  beam.ParDo(AllDoFn()) | "Write" >> WriteToText("gs://" + bucket_name + "/output/output")

