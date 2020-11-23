import csv
import os
import sys
import pandas as pd
import numpy as np

import stellargraph as sg

from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import GCNSupervisedGraphClassification
from sklearn import model_selection
from IPython.display import display, HTML

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt
from stellargraph import datasets

correct_data_dir = sys.argv[1]
incorrect_data_dir = sys.argv[2]

def create_graph_from_edges(edges_file):
  with open(edges_file, 'r') as f:
    lines = f.readlines()
    sources = []
    targets = []
    for line in lines:
      if len(line.split(" ")) < 2:
      	continue
      source = int(line.split(" ")[0].strip())
      target = int(line.split(" ")[1].strip())
      sources.append(source)
      targets.append(target)
  edges = pd.DataFrame({"source": sources, "target": targets})
  
  nodes = []
  for item in sources+targets:
    if item not in nodes:
      nodes.append(item)
  
  node_degrees = {}
  for node in nodes:
    node_degrees[node] = 1
  for node in nodes:
    node_degrees[node] += 1
  
  nodes = pd.DataFrame({"d": node_degrees}, index=nodes)

  return sg.StellarGraph(nodes, edges)


def train(graphs, graph_labels, epochs=200, folds=10, n_repeats=5):
  graph_labels.value_counts().to_frame()
  graph_labels = pd.get_dummies(graph_labels, drop_first=True)
  generator = PaddedGraphGenerator(graphs=graphs)

  def create_graph_classification_model(generator):
    gc_model = GCNSupervisedGraphClassification(
        layer_sizes=[64, 64],
        activations=["relu", "relu"],
        generator=generator,
        dropout=0.5,
    )
    x_inp, x_out = gc_model.in_out_tensors()
    predictions = Dense(units=32, activation="relu")(x_out)
    predictions = Dense(units=16, activation="relu")(predictions)
    predictions = Dense(units=1, activation="sigmoid")(predictions)

    # Let's create the Keras model and prepare it for training
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(0.0005), loss=binary_crossentropy, metrics=["acc"])

    return model

  es = EarlyStopping(
    monitor="loss", min_delta=0, patience=25, restore_best_weights=True
  )

  def train_fold(model, train_gen, test_gen, es, epochs):
    history = model.fit(
        train_gen, epochs=epochs, validation_data=test_gen, verbose=0, callbacks=[es],
    )
    # calculate performance on the test data and return along with history
    test_metrics = model.evaluate(test_gen, verbose=0)
    test_acc = test_metrics[model.metrics_names.index("acc")]

    return history, test_acc

  def get_generators(train_index, test_index, graph_labels, batch_size):
    train_gen = generator.flow(
        train_index, targets=graph_labels.iloc[train_index].values, batch_size=batch_size
    )
    test_gen = generator.flow(
        test_index, targets=graph_labels.iloc[test_index].values, batch_size=batch_size
    )

    return train_gen, test_gen

  test_accs = []

  stratified_folds = model_selection.RepeatedStratifiedKFold(
    n_splits=folds, n_repeats=n_repeats).split(graph_labels, graph_labels)

  max_acc = 0
  for i, (train_index, test_index) in enumerate(stratified_folds):
    print(f"Training and evaluating on fold {i+1} out of {folds * n_repeats}...")
    train_gen, test_gen = get_generators(train_index, test_index, graph_labels, batch_size=5)
    model = create_graph_classification_model(generator)
    history, acc = train_fold(model, train_gen, test_gen, es, epochs)
    test_accs.append(acc)
    print(acc)
    if acc > max_acc:
      tf.keras.models.save_model(model, "model", save_format='tf')
      max_acc = acc
  
  print("Maximum accuracy is: " + str(max_acc*100) + "%")


if __name__ == '__main__':
  # prepare the data
  graphs = []
  graph_labels = []

  print("~~~ Preparing the data ~~~")

  index = 0
  for f in os.listdir(correct_data_dir):
    index += 1
    if index == 25:
      break
    graphs.append(create_graph_from_edges(correct_data_dir+'/'+f))
    graph_labels.append(1)

  index = 0
  for f in os.listdir(incorrect_data_dir):
    index += 1
    if index  == 25:
      break
    graphs.append(create_graph_from_edges(incorrect_data_dir+'/'+f))
    graph_labels.append(-1)
  
  graph_labels = pd.Index(graph_labels)

  print("~~~ Training ~~~")

  # train
  train(graphs, graph_labels)
