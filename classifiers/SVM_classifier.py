import networkx as nx
import sys
import os

from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

correct_data_dir = sys.argv[1]
incorrect_data_dir = sys.argv[2]

TRAIN = 100
TEST = 65
EDGE_LIMIT = 50000

def create_graph_from_edges(edges_file, label):
  G = nx.Graph()

  with open(edges_file, 'r') as f:
    lines = f.readlines()
    if len(lines) > EDGE_LIMIT:
      return None
    sources = []
    targets = []

    for line in lines:
      if len(line.split(" ")) < 2:
      	continue
      
      source = str(line.split(" ")[0].strip())
      target = str(line.split(" ")[1].strip())
      
      sources.append(source)
      targets.append(target)
  
  
  for i in range(len(sources)):
    G.add_edge(sources[i], targets[i])

  for node in G.nodes:
    G.nodes[node]['label'] = label

  return G


def train(train_graphs, test_graphs, train_labels, test_labels):
  G_train = list(graph_from_networkx(train_graphs, node_labels_tag='label'))
  G_test = list(graph_from_networkx(test_graphs, node_labels_tag='label'))

  gk = WeisfeilerLehman(n_iter=5, normalize=False, base_graph_kernel=VertexHistogram)
  
  # Construct kernel matrices
  K_train = gk.fit_transform(G_train)
  K_test = gk.transform(G_test)
  
  # Train an SVM classifier and make predictions
  clf = SVC(kernel='poly')
  clf.fit(K_train, train_labels)
  
  pred_labels = clf.predict(K_test)
  print("Train data: " + str(TRAIN))
  print("Test data: " + str(TEST))
  print("Accuracy: " + str(accuracy_score(pred_labels, test_labels)))
  

if __name__ == '__main__':
  
  correct_graphs, correct_labels = [], []
  incorrect_graphs, incorrect_labels = [], []

  print("~~~ Preparing the training data ~~~")

  index  = 0
  for f in os.listdir(correct_data_dir): 
    res = create_graph_from_edges(correct_data_dir+'/'+f, '1')
    if res:
      correct_graphs.append(res)
      correct_labels.append('1')
    index += 1
    if index == TEST+TRAIN:
      break
 
  index = 0
  for f in os.listdir(incorrect_data_dir):
    res = create_graph_from_edges(incorrect_data_dir+'/'+f, '-1')
    if res:
      incorrect_graphs.append(res)
      incorrect_labels.append('-1')
    index += 1
    if index == TEST+TRAIN:
      break

  train_low_index = 0
  train_high_index = int(TRAIN/2)
  train_graphs = correct_graphs[train_low_index:train_high_index] + incorrect_graphs[train_low_index:train_high_index]
  train_labels = correct_labels[train_low_index:train_high_index] + incorrect_labels[train_low_index:train_high_index]
  
  test_low_index = int(TRAIN/2)
  test_high_index = int(TRAIN/2)+int(TEST/2)
  test_graphs = correct_graphs[test_low_index:test_high_index] + incorrect_graphs[test_low_index:test_high_index]
  test_labels = correct_labels[test_low_index:test_high_index] + incorrect_labels[test_low_index:test_high_index]

  print("~~~ Training ~~~")

  train(train_graphs, test_graphs, train_labels, test_labels)

