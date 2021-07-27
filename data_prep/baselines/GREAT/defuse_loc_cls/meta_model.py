import tensorflow as tf
from models import great_transformer, ggnn, rnn, util

class DefuseLocModel(tf.keras.layers.Layer):
  def __init__(self, config, vocab_dim):
    super(DefuseLocModel, self).__init__()
    self.config = config
    self.vocab_dim = vocab_dim

  def build(self, _):
    # These layers are always used; initialize with any given model's hidden_dim
    random_init = tf.random_normal_initializer(stddev=self.config['base']['hidden_dim'] ** -0.5)
    self.embed = tf.Variable(random_init([self.vocab_dim, self.config['base']['hidden_dim']]), dtype=tf.float32)
    self.prediction = tf.keras.layers.Dense(1)
    self.pos_enc = tf.constant(util.positional_encoding(self.config['base']['hidden_dim'], 5000))

    join_dicts = lambda d1, d2: {**d1, **d2}
    base_config = self.config['base']
    desc = self.config['configuration'].split(' ')
    self.stack = []
    for kind in desc:
      if kind == 'great':
        self.stack.append(great_transformer.Transformer(join_dicts(self.config['transformer'], base_config), shared_embedding=self.embed))
      else:
        raise ValueError('Unknown model component provided:', kind)

  def call(self, tokens, token_mask, edges, training):
    subtoken_embeddings = tf.nn.embedding_lookup(self.embed, tokens)
    subtoken_embeddings *= tf.expand_dims(tf.cast(tf.clip_by_value(tokens, 0, 1), dtype='float32'), -1)
    states = tf.reduce_mean(subtoken_embeddings, 2)
    if not self.stack or not isinstance(self.stack[0], rnn.RNN):
      states += self.pos_enc[:tf.shape(states)[1]]

    for model in self.stack:
      if isinstance(model, great_transformer.Transformer):
        mask = tf.cast(token_mask, dtype='float32')
        mask = tf.expand_dims(tf.expand_dims(mask, 1), 1)
        attention_bias = tf.stack([edges[:, 0], edges[:, 1], edges[:, 3], edges[:, 2]], axis=1)
        states = model(states, mask, attention_bias, training=training)
      else:
        raise ValueError('Model not yet supported:', model)

    return tf.transpose(self.prediction(states), [0, 2, 1])

  def get_loss(self, logits, token_mask, labels, locations):
    seq_mask = tf.cast(token_mask, 'float32')
    logits += (1.0 - tf.expand_dims(seq_mask, 1)) * tf.float32.min
    logits = tf.sigmoid(logits)

    probs = [logits[idx, :, loc-1] for idx, loc in enumerate(locations)]
    probs = tf.reshape(probs, [len(probs), 1])

    labels = tf.cast(labels, 'float32')
    loss = -labels * tf.math.log(probs + 1e-9) - (1.0 - labels) * tf.math.log(1.0 - probs + 1e-9)
    loss = tf.reduce_mean(loss)

    pred_labels = tf.round(probs)
    acc = tf.reduce_mean(tf.cast(tf.math.equal(pred_labels, labels), 'float32'))
    auc = tf.keras.metrics.AUC()
    auc.update_state(tf.cast(labels, 'float32'), probs)
    return loss, acc, auc.result().numpy()

  def run_dummy_input(self):
    self(tf.ones((3,3,10), dtype='int32'), tf.ones((3,3), dtype='int32'), tf.ones((2,4), dtype='int32'), True)
