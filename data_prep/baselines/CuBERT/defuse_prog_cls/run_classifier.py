import collections
import json
import sys
import os

import tensorflow as tf

import model.modeling as modeling
import model.optimization as optimization
import tokenizing.tokenization as tokenization
import tokenizing.python_tokenizer as python_tokenizer
import tokenizing.unified_tokenizer as unified_tokenizer


MAX_OUTPUT_TOKEN_LENGTH = 15 

## Required parameters
mode = sys.argv[1]
data_dir = sys.argv[2]
output_dir = sys.argv[3]
init_checkpoint = sys.argv[4]

## Other parameters
vocab_file = "vocab.txt"
bert_config_file = "bert_config.json"
do_lower_case = True
max_seq_length = 512
train_batch_size = 32
eval_batch_size = 8
predict_batch_size = 2
learning_rate = 1e-5
num_train_epochs = 3
warmup_proportion = 0.1
save_checkpoints_steps = 1000
iterations_per_loop = 1000
use_tpu = False
tpu_name = None
tpu_zone = None
gcp_project = None
num_tpu_cores = 8

do_train = True if mode == 'TRAIN' else False
do_eval = True if mode == 'EVAL' else False


class InputExample(object):
  def __init__(self, guid, function, label=None):
    self.guid = guid
    self.function = function
    self.label = label


class PaddingInputExample(object):
  pass


class InputFeatures(object):
  def __init__(self, input_ids, input_mask, segment_ids, label_id, is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


class DataProcessor(object):
  def get_train_examples(self, data_dir):
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    raise NotImplementedError()

  def get_eval_examples(self, data_dir):
    raise NotImplementedError()

  def get_labels(self):
    raise NotImplementedError()

  @classmethod
  def _read_json(cls, input_file, quotechar=None):
    with tf.compat.v1.gfile.GFile(input_file) as f:
      lines = f.readlines()
      functions = []
      labels = []
      
      data_lines = []
      for line in lines:
        jline = json.loads(str(line))
        data_lines.append((jline['function'], jline['label']))

      return data_lines

class DefuseProgProcessor(DataProcessor):
  def get_train_examples(self, data_dir):
    data_tuples = []
    for f in os.listdir(os.path.join(data_dir, 'train')):
      data_tuples += self._read_json(os.path.join(data_dir, 'train', f))
    examples = []
    for (i, data_item) in enumerate(data_tuples):
      guid = "train-%d" % (i)
      function = tokenization.convert_to_unicode("[CLS]" + data_item[0])
      label = tokenization.convert_to_unicode(data_item[1])
      examples.append(InputExample(guid=guid, function=function, label=label))
    print("~~~~~ end of get train examples..")
    return examples

  def get_dev_examples(self, data_dir):
    data_tuples = []
    for f in os.listdir(os.path.join(data_dir, 'dev')):
      data_tuples += self._read_json(os.path.join(data_dir, 'dev', f))
    examples = []
    for (i, data_item) in enumerate(data_tuples):
      guid = "dev-%d" % (i)
      function = tokenization.convert_to_unicode("[CLS]" + data_item[0])
      label = tokenization.convert_to_unicode(data_item[1])
      examples.append(InputExample(guid=guid, function=function, label=label))
    return examples

  def get_eval_examples(self, data_dir):
    data_tuples = []
    for f in os.listdir(os.path.join(data_dir, 'eval')):
      data_tuples += self._read_json(os.path.join(data_dir, 'eval', f))
    data_tuples = self._read_json(os.path.join(data_dir, 'eval', "eval.txt"))
    examples = []
    for (i, data_item) in enumerate(data_tuples):
      guid = "eval-%d" % (i)
      function = tokenization.convert_to_unicode("[CLS]" + data_item[0])
      label = tokenization.convert_to_unicode(data_item[1])
      examples.append(InputExample(guid=guid, function=function, label=label))
    return examples

  def get_labels(self):
    return ['used', 'unused']


def convert_single_example(index, example, label_list, seq_length, tokenizer):
  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * seq_length,
        input_mask=[0] * seq_length,
        segment_ids=[0] * seq_length,
        label_id=0,
        is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  function_sentences = tokenizer.tokenize(example.function)
  
  function_int = []
  for sentence in function_sentences:
    function_int += sentence

  if len(function_int) > seq_length:
    function_int = function_int[0:seq_length]

  segment_ids = [0] * len(function_int)
  input_ids = function_int
  input_mask = [1] * len(input_ids)
  while len(input_ids) < seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == seq_length
  assert len(input_mask) == seq_length
  assert len(segment_ids) == seq_length

  label_id = label_map[example.label]

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  writer = tf.io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f
    
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  name_to_features = {
      "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.io.FixedLenFeature([], tf.int64),
      "is_real_example": tf.io.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, 'int32')
      example[name] = t

    return example

  def input_fn(params):
    batch_size = params["batch_size"]

    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    return d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

  return input_fn


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()
  hidden_size = output_layer.shape[-1]

  output_weights = tf.compat.v1.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.compat.v1.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.compat.v1.variable_scope("loss"):
    if is_training:
      output_layer = tf.nn.dropout(output_layer, rate=0.1)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    tf.compat.v1.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.compat.v1.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None

    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.compat.v1.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names)\
              = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      
      if use_tpu:
        def tpu_scaffold():
          tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.compat.v1.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.compat.v1.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.compat.v1.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.compat.v1.metrics.mean(values=per_example_loss, weights=is_real_example)
        norm_logits = (logits[:,-1] - tf.math.reduce_min(logits[:,-1])) / (tf.math.reduce_max(logits[:,-1]) - tf.math.reduce_min(logits[:,-1]))
        auc = tf.compat.v1.metrics.auc(label_ids, norm_logits)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
            "eval_auc": auc
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn

@tf.function
def main():
  processor = DefuseProgProcessor()
  tokenization.validate_case_matches_checkpoint(do_lower_case, init_checkpoint)
  bert_config = modeling.BertConfig.from_json_file(bert_config_file)
  tf.compat.v1.gfile.MakeDirs(output_dir)

  label_list = processor.get_labels();
  tokenizer = python_tokenizer.PythonTokenizer(
      max_output_token_length = MAX_OUTPUT_TOKEN_LENGTH,
      vocab_path=vocab_file)

  tpu_cluster_resolver = None
  if use_tpu and tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        tpu_name, zone=tpu_zone, project=gcp_project)

  is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.compat.v1.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=output_dir,
      save_checkpoints_steps=save_checkpoints_steps,
      tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
          iterations_per_loop=iterations_per_loop,
          num_shards=num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None

  if do_train:
    print(" -- - - - TRAINING ----- --")
    train_examples = processor.get_train_examples(data_dir)
    assert len(train_examples) > 0, "number of train examples must be nonzero"
    print("GOT TRAINING EXAMPLES: ", len(train_examples), "examples.")
    num_train_steps = int(len(train_examples) / train_batch_size * num_train_epochs)
    num_warmup_steps = int(num_train_steps * warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=init_checkpoint,
      learning_rate=learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=use_tpu,
      use_one_hot_embeddings=use_tpu)

  estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
      use_tpu=use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=train_batch_size,
      eval_batch_size=eval_batch_size,
      predict_batch_size=predict_batch_size)

  if do_train:
    train_file = os.path.join(output_dir, "train.tf_record")
    print(" ---- CONVERTING TO FEATURES....")
    file_based_convert_examples_to_features(
        train_examples, label_list, max_seq_length, tokenizer, train_file)
    tf.compat.v1.logging.info("***** Running training *****")
    tf.compat.v1.logging.info("  Num examples = %d", len(train_examples))
    tf.compat.v1.logging.info("  Batch size = %d", train_batch_size)
    tf.compat.v1.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if do_eval:
    eval_examples = processor.get_eval_examples(data_dir)
    num_actual_eval_examples = len(eval_examples)
    assert num_actual_eval_examples > 0, "number of evaluation examples must be nonzero"
    if use_tpu:
      while len(eval_examples) % eval_batch_size != 0:
        eval_examples.append(PaddingInputExample())

    eval_file = os.path.join(output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(eval_examples, label_list, max_seq_length, tokenizer, eval_file)

    tf.compat.v1.logging.info("***** Running evaluation *****")
    tf.compat.v1.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.compat.v1.logging.info("  Batch size = %d", eval_batch_size)

    eval_steps = None
    if use_tpu:
      assert len(eval_examples) % eval_batch_size == 0
      eval_steps = int(len(eval_examples) // eval_batch_size)

    eval_drop_remainder = True if use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    with tf.compat.v1.gfile.GFile(output_eval_file, "w") as writer:
      tf.compat.v1.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.compat.v1.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
  main()
