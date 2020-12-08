import networkx as nx
import sys
import os

from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.svm import SVC

data_dir = sys.argv[1]

TRAIN = 300
TEST = 100
EDGE_LIMIT = 20000


def create_graph_from_edges(edges_file, counter, label):
  G = nx.Graph()

  with open(edges_file, 'r') as f:
    lines = f.readlines()
    sources = []
    targets = []
    if len(lines) > EDGE_LIMIT:
      print("SKIPPED >>> "+str(len(lines)))
      return None

    for line in lines:
      if len(line.split(" ")) < 2:
      	continue
      
      source = int(line.split(" ")[0].strip())
      target = int(line.split(" ")[1].strip())
      
      sources.append(source)
      targets.append(target)
  
  
  for i in range(len(sources)):
    G.add_edge(sources[i], targets[i])

  for node in G.nodes:
    G.nodes[node]['label'] = 0

  return G


def train_with_kernel(K_train, K_test, train_labels, test_labels, kernel):
  print(kernel)
  clf = SVC(gamma='auto', kernel=kernel)
  clf.fit(K_train, train_labels)
  pred_labels = clf.predict(K_test)

  return pred_labels

def train(train_graphs, test_graphs, train_labels, test_labels):
  G_train = list(graph_from_networkx(train_graphs, node_labels_tag='label'))
  G_test = list(graph_from_networkx(test_graphs, node_labels_tag='label'))
  
  print(" ~~~ WeisfeilerLehman ~~~")
  gk = WeisfeilerLehman(n_iter=5, normalize=False, base_graph_kernel=VertexHistogram)
  
  K_train = gk.fit_transform(G_train)
  K_test = gk.transform(G_test)
  
  pred_labels = train_with_kernel(K_train, K_test, train_labels, test_labels, 'precomputed')
  print("Train data: " + str(TRAIN))
  print("Test data: " + str(TEST))
  print("Accuracy: " + str(accuracy_score(pred_labels, test_labels)))


def prepare_data(data_dir, mode, threshold):
  # mode = test, train
  edges, labels = [], []
  graphs = []
  graphs_dict = {}
  with open(data_dir+'/'+mode+'/'+'labels.txt') as f:
    lines = f.readlines()
    for line in lines:
      filename = "graph-"+line.strip().split(" ")[0]+".py.edges"
      labelname = line.strip().split(" ")[1]
      graphs_dict[filename] = labelname

  index = 0
  for f in os.listdir(data_dir+'/'+mode):
    if f.endswith(".txt"):
      continue
    
    graph = create_graph_from_edges(data_dir+'/'+mode+'/'+f, index, graphs_dict[f])
    if graph:
      graphs.append(graph)
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

