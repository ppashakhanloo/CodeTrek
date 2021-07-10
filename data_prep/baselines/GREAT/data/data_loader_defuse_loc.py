import os
import json
import random
import tensorflow as tf


EDGE_TYPES = {
  'enum_CFG_NEXT': 0,
  'enum_LAST_READ': 1,
  'enum_LAST_WRITE': 2,
  'enum_COMPUTED_FROM': 3,
  'enum_RETURNS_TO': 4,
  'enum_FORMAL_ARG_NAME': 5,
  'enum_FIELD': 6,
  'enum_SYNTAX': 7,
  'enum_NEXT_SYNTAX': 8,
  'enum_LAST_LEXICAL_USE': 9,
  'enum_CALLS': 10
}

class DataLoader():

  def __init__(self, data_path, data_config, vocabulary):
    self.data_path = data_path
    self.config = data_config
    self.vocabulary = vocabulary

  def batcher(self, mode="train"):
    data_path = self.get_data_path(mode)
    dataset = tf.data.Dataset.list_files(data_path + '/*.txt*', shuffle=mode != 'eval', seed=42)
    dataset = dataset.interleave(lambda x: tf.data.TextLineDataset(x).shuffle(buffer_size=1000) if mode == 'train' else tf.data.TextLineDataset(x), cycle_length=4, block_length=16)
    dataset = dataset.prefetch(1)
    if mode == "train":
      dataset = dataset.repeat()

    ds = tf.data.Dataset.from_generator(lambda mode: self.to_batch(dataset, mode), output_types=(tf.dtypes.int32, tf.dtypes.int32, tf.dtypes.int32, tf.dtypes.int32, tf.dtypes.int32), args=(mode,))
    ds = ds.prefetch(1)
    return ds

  def get_data_path(self, mode):  
    if mode == "train":
      return os.path.join(self.data_path, "train")
    elif mode == "dev":
      return os.path.join(self.data_path, "dev")
    elif mode == "eval":
      return os.path.join(self.data_path, "eval")
    else:
      raise ValueError("Mode % not supported for batching; please use \"train\", \"dev\", or \"eval\".")

  def to_sample(self, json_data):
    def parse_edges(edges):
      # Reorder edges to [edge type, source, target] and double edge type index to allow reverse edges
      relations = [[2*EDGE_TYPES[rel[3]], rel[0], rel[1]] for rel in edges if rel[3] in EDGE_TYPES]
      relations += [[rel[0] + 1, rel[2], rel[1]] for rel in relations]  # Add reverse edges
      return relations

    edges = parse_edges(json_data["edges"])
    flat_edges = []
    for e in edges:
      flat_edges.append(e[1])
      flat_edges.append(e[2])
    padding = max(0, max(flat_edges) - len(json_data['source_tokens']))
    tokens = [self.vocabulary.translate(t)[:self.config["max_token_length"]] \
              for t in ['[CLS]'] + padding * [''] + json_data["source_tokens"]]
    label = 0 if json_data["label"] == 'used' else 1
    location = json_data["error_location"]
    return (tokens, edges, label, location, [])

  def to_batch(self, sample_generator, mode):
    if isinstance(mode, bytes): mode = mode.decode('utf-8')
    def sample_len(sample):
      return len(sample[0])

    def make_batch(buffer):
      pivot = sample_len(random.choice(buffer))
      buffer = sorted(buffer, key=lambda b: abs(sample_len(b) - pivot))
      batch = []
      max_seq_len = 0
      for sample in buffer:
        max_seq_len = max(max_seq_len, sample_len(sample))
        if max_seq_len*(len(batch) + 1) > self.config['max_batch_size']:
          break
        batch.append(sample)
      batch_dim = len(batch)
      buffer = buffer[batch_dim:]
      batch = list(zip(*batch))

      token_tensor = tf.ragged.constant(batch[0], dtype=tf.dtypes.int32).\
              to_tensor(shape=(len(batch[0]), max(len(b) for b in batch[0]), self.config["max_token_length"]))

      label = tf.constant(batch[2], dtype=tf.dtypes.int32)
      edge_batches = tf.repeat(tf.range(batch_dim), [len(edges) for edges in batch[1]])
      edge_tensor = tf.concat(batch[1], axis=0)
      edge_tensor = tf.stack([edge_tensor[:, 0], edge_batches, edge_tensor[:, 1], edge_tensor[:, 2]], axis=1)
      #t_batches = tf.repeat(tf.range(batch_dim), [len(t) for t in batch[3]])
      location = tf.constant(batch[3], dtype=tf.dtypes.int32)
      
      #items1 = tf.cast(tf.concat(batch[3], axis=0), 'int32')
      #items1 = tf.stack([t_batches, items1], axis=1)
      c_batches = tf.repeat(tf.range(batch_dim), [len(c) for c in batch[4]])
      items2 = tf.cast(tf.concat(batch[4], axis=0), 'int32')
      items2 = tf.stack([c_batches, items2], axis=1)
      return buffer, (token_tensor, edge_tensor, label, location, items2)

    buffer = []
    num_samples = 0
    for line in sample_generator:
      json_sample = json.loads(line.numpy())
      sample = self.to_sample(json_sample)
      if sample_len(sample) > self.config['max_sequence_length']:
        continue
      buffer.append(sample)
      num_samples += 1
      if mode == 'dev' and num_samples >= self.config['max_valid_samples']:
        break
      if sum(sample_len(sample) for l in buffer) > self.config['max_buffer_size']*self.config['max_batch_size']:
        buffer, batch = make_batch(buffer)
        yield batch
    while buffer:
      buffer, batch = make_batch(buffer)
      yield batch
