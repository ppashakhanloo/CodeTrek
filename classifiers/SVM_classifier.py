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

TRAIN = 500
TEST = 300

def create_graph_from_edges(edges_file, counter):
  G = nx.Graph()

  with open(edges_file, 'r') as f:
    lines = f.readlines()
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

  node_degrees = {}
  for node in G.nodes():
    node_degrees[node] = 1
  #for node in G.nodes():
  #  node_degrees[node] += 1

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
  
  # Train an SVM classifier and make predictions
  clf = SVC(gamma='auto')
  clf.fit(K_train, train_labels)
  pred_labels = clf.predict(K_test)
  
  print(accuracy_score(pred_labels, test_labels))
  


if __name__ == '__main__':
  # prepare the data
  test_graphs = []
  test_labels = []
  train_graphs = []
  train_labels = []
  
  corr_graphs = []
  corr_labels = []
  incorr_graphs = []
  incorr_labels = []

  print("~~~ Preparing the training data ~~~")

  index = 0
  for f in os.listdir(correct_data_dir):
    corr_graphs.append(create_graph_from_edges(correct_data_dir+'/'+f, index))
    corr_labels.append(1)
    index += 1
    if index == (TEST+TRAIN)/2+1:
      break
 
  index = 0
  for f in os.listdir(incorrect_data_dir):
    incorr_graphs.append(create_graph_from_edges(incorrect_data_dir+'/'+f, index))
    incorr_labels.append(-1)
    index += 1
    if index == (TEST+TRAIN)/2+1:
      break
  
  train_graphs = corr_graphs[0:int(TRAIN/2)] + incorr_graphs[0:int(TRAIN/2)]
  test_graphs = corr_graphs[int(TRAIN/2)+1:] + incorr_graphs[int(TRAIN/2)+1:]

  train_labels = corr_labels[0:int(TRAIN/2)] + incorr_labels[0:int(TRAIN/2)]
  test_labels = corr_labels[int(TRAIN/2)+1:] + incorr_labels[int(TRAIN/2)+1:]

  print("~~~ Training ~~~")

  train(train_graphs, test_graphs, train_labels, test_labels)

