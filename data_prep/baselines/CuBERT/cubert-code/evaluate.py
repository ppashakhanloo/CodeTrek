import os, sys, json, glob
import numpy as np

from sklearn.metrics import roc_auc_score


with open('output_results/test_results.tsv', 'r') as f:
  lines = f.readlines()
  all_scores = []
  for line in lines:
    line = line.strip().replace('\t', ' ').replace(' ', ' ').replace('  ', ' ').split(' ')
    all_scores.append([float(line[0]), float(line[1])])
 
all_scores = np.array(all_scores)

test_path = '/home/pardisp/cubert-code/data/codesearch/eval.jsontxt-0'
with open(test_path, 'r') as f:
    lines = f.readlines()
    real_labels = []
    for l in lines:
        d = json.loads(l.strip())
        real_labels.append(0 if d['label'] == 'Correct' else 1)

#print(all_scores.shape)
#print(real_labels)

#print(roc_auc_score(real_labels, all_scores))
  
all_scores = np.argmax(np.array(all_scores), -1)
print(all_scores)
