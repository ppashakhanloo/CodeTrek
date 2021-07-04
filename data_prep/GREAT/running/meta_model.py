import tensorflow as tf
from models import great_transformer, ggnn, rnn, util

class DefuseProgModel(tf.keras.layers.Layer):
  def __init__(self, config, vocab_dim):
    super(DefuseProgModel, self).__init__()
    self.config = config
    self.vocab_dim = vocab_dim

  def build(self, _):
    # These layers are always used; initialize with any given model's hidden_dim
    random_init = tf.random_normal_initializer(stddev=self.config['base']['hidden_dim'] ** -0.5)
    self.embed = tf.Variable(random_init([self.vocab_dim, self.config['base']['hidden_dim']]), dtype=tf.float32)
    self.prediction = tf.keras.layers.Dense(1)

    # Store for convenience
    self.pos_enc = tf.constant(util.positional_encoding(self.config['base']['hidden_dim'], 5000))

    # Next, parse the main 'model' from the config
    join_dicts = lambda d1, d2: {**d1, **d2}  # Small util function to combine configs
    base_config = self.config['base']
    desc = self.config['configuration'].split(' ')
    self.stack = []
    for kind in desc:
      if kind == 'rnn':
        self.stack.append(rnn.RNN(join_dicts(self.config['rnn'], base_config), shared_embedding=self.embed))
      elif kind == 'ggnn':
        self.stack.append(ggnn.GGNN(join_dicts(self.config['ggnn'], base_config), shared_embedding=self.embed))
      elif kind == 'great':
        self.stack.append(great_transformer.Transformer(join_dicts(self.config['transformer'], base_config), shared_embedding=self.embed))
      elif kind == 'transformer':  # Same as above, but explicitly without bias_dim set -- defaults to regular Transformer.
        joint_config = join_dicts(self.config['transformer'], base_config)
        joint_config['num_edge_types'] = None
        self.stack.append(great_transformer.Transformer(joint_config, shared_embedding=self.embed))
      else:
        raise ValueError('Unknown model component provided:', kind)

  def call(self, tokens, token_mask, edges, training):
    # Embed subtokens and average into token-level embeddings, masking out invalid locations
    subtoken_embeddings = tf.nn.embedding_lookup(self.embed, tokens)
    subtoken_embeddings *= tf.expand_dims(tf.cast(tf.clip_by_value(tokens, 0, 1), dtype='float32'), -1)
    states = tf.reduce_mean(subtoken_embeddings, 2)
    if not self.stack or not isinstance(self.stack[0], rnn.RNN):
      states += self.pos_enc[:tf.shape(states)[1]]

    for model in self.stack:
      if isinstance(model, rnn.RNN):  # RNNs simply use the states
        states = model(states, training=training)
      elif isinstance(model, ggnn.GGNN):  # For GGNNs, pass edges as-is
        states = model(states, edges, training=training)
      elif isinstance(model, great_transformer.Transformer):  # For Transformers, reverse edge directions to match query-key direction and add attention mask.
        mask = tf.cast(token_mask, dtype='float32')
        mask = tf.expand_dims(tf.expand_dims(mask, 1), 1)
        attention_bias = tf.stack([edges[:, 0], edges[:, 1], edges[:, 3], edges[:, 2]], axis=1)
        states = model(states, mask, attention_bias, training=training)
      else:
        raise ValueError('Model not yet supported:', model)

    return tf.transpose(self.prediction(states), [0, 2, 1])

  def get_loss(self, logits, token_mask, labels, items1, items2):
    seq_mask = tf.cast(token_mask, 'float32')
    logits += (1.0 - tf.expand_dims(seq_mask, 1)) * tf.float32.min
    probs = tf.clip_by_value(logits[:,0], 0, 1)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, probs)
    acc = tf.keras.metrics.sparse_categorical_accuracy(labels, probs)
    auc = tf.keras.metrics.AUC()
    auc.update_state(labels, tf.reduce_mean(probs,1))
    return loss, acc, auc.result().numpy()

  def run_dummy_input(self):
    self(tf.ones((3,3,10), dtype='int32'), tf.ones((3,3), dtype='int32'), tf.ones((2,4), dtype='int32'), True)
