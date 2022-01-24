import sys
import os
import json

if len(sys.argv) == 1:
  print("python exec.py code_file doc_file label")
  sys.exit(0)

code_file = sys.argv[1]
doc_file = sys.argv[2]
label = sys.argv[3]

with open(code_file, 'r') as f:
  code = f.read()

with open(doc_file, 'r') as f:
  doc = f.read()

sample = {
            'function': code,
            'docstring': doc,
            'label': label,
            'info': "unknown"
         }

with open('data/function_docstring_datasets_small/eval.jsontxt-00003-of-00004', 'w') as f:
  json.dump(sample, f)

os.system("python3 cubert/run_classifier.py --do_train=False --bert_config_file=models/bert_large_config.json --vocab_file=data/github_python_minus_ethpy150open_deduplicated_vocabulary.txt --task_name=docstring --init_checkpoint=models/model.ckpt-6072 --data_dir=data/function_docstring_datasets_small --output_dir=docstring_results --do_eval=False --do_predict=True 2> /dev/null")

print("\nCorrect\t\tIncorrect\n")
os.system("cat docstring_results/test_results.tsv")
