import networkx as nx
import sys
import os

from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

data_path = sys.argv[1]

TRAIN = 1000
TEST = 300

def create_graph_from_edges(edges_file, counter):
  G = nx.Graph()

  with open(edges_file, 'r') as f:
    lines = f.readlines()
    sources = []
    targets = []

    for line in lines:
      if len(line.strip().split(" ").strip()) < 2:
      	continue
      
      source = str(line.split(" ")[0].strip())
      target = str(line.split(" ")[1].strip())
      
      sources.append(source)
      targets.append(target)
  
  
  for i in range(len(sources)):
    G.add_edge(sources[i], targets[i])

  node_degrees = {}
  for node in G.nodes():
    node_degrees[node] = 1

  for node in node_degrees:
    G.nodes[node]['label'] = node_degrees[node]

  return G


def train(train_graphs, test_graphs, train_labels, test_labels):
  G_train = list(graph_from_networkx(train_graphs, node_labels_tag='label'))
  G_test = list(graph_from_networkx(test_graphs, node_labels_tag='label'))

  gk = WeisfeilerLehman(n_iter=5, normalize=False, base_graph_kernel=VertexHistogram)
  
  # Construct kernel matrices
  K_train = gk.fit_transform(G_train)
  K_test = gk.transform(G_test)
  
  print(K_train)
  # Train an SVM classifier and make predictions
  clf = SVC(gamma='auto')
  clf.fit(K_train, train_labels)
  pred_labels = clf.predict(K_test)
  
  print(accuracy_score(pred_labels, test_labels))
  

def prepare_data(data_dir, mode, threshold):
  # mode = test, train
  edges, labels = [], []
  graphs = []

  graphs_dict = {}
  with open(data_dir+'/'+mode+'/'+'labels.txt') as f:
    lines = l.readlines()
    for line in lines:
      filename = "graph-"+line.strip().split(" ")[0]+".py"
      labelname = line.strip().split(" ")[1]
      graphs_dict[filename] = labelname

  index = 0
  for f in os.listdir(data_dir+'/'+mode):
    graphs.append(create_graph_from_edges(data_dir+'/'+mode+'/'+f, index))
    labels.append(graphs_dict[f])
    index += 1
    if index == threshold:
      break

  return graphs, labels

if __name__ == '__main__':
  
  print("~~~ Preparing the training data ~~~")

  train_graphs, train_labels = prepare_data(data_dir, 'train', TRAIN)
  test_graphs, test_labels = prepare_data(data_dir, 'test', TEST)

  print("~~~ Training ~~~")

  train(train_graphs, test_graphs, train_labels, test_labels)

