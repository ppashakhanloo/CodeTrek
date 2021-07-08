import sys
sys.path.append('.')

import argparse
import yaml

import numpy as np
import tensorflow as tf

from checkpoint_tracker import Tracker
from data import data_loader_exception, vocabulary
from meta_model import ExceptionClsModel

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("data_path", help="Path to data root")
  ap.add_argument("vocabulary_path", help="Path to vocabulary file")
  ap.add_argument("config", help="Path to config file")
  ap.add_argument("-m", "--models", help="Directory to store trained models (optional)")
  ap.add_argument("-l", "--log", help="Path to store training log (optional)")
  ap.add_argument("-e", "--eval_only", help="Whether to run just the final model evaluation")
  args = ap.parse_args()
  config = yaml.safe_load(open(args.config))
  print("Training with configuration:", config)
  data = data_loader_exception.DataLoader(args.data_path, config["data"], vocabulary.Vocabulary(args.vocabulary_path))
  if args.eval_only:
    if args.models is None or args.log is None:
      raise ValueError("Must provide a path to pre-trained models when running final evaluation")
    test(data, config, args.models, args.log)
  else:
    train(data, config, args.models, args.log)

def test(data, config, model_path, log_path):
  model = ExceptionClsModel(config['model'], data.vocabulary.vocab_dim)
  model.run_dummy_input()
  tracker = Tracker(model, model_path, log_path)
  tracker.restore(best_model=True)
  acc, auc = evaluate(data, config, model, is_heldout=False)
  print("Final Results: acc: {0:.2%}, auc: {1:.2%}".format(acc, auc))

def train(data, config, model_path=None, log_path=None):
  model = ExceptionClsModel(config['model'], data.vocabulary.vocab_dim)
  model.run_dummy_input()
  print("Model initialized, training {:,} parameters".format(np.sum([np.prod(v.shape) for v in model.trainable_variables])))
  optimizer = tf.optimizers.Adam(config["training"]["learning_rate"])

  # Restore model from checkpoints if present; also sets up logger
  if model_path is None:
    tracker = Tracker(model)
  else:
    tracker = Tracker(model, model_path, log_path)
  tracker.restore()
  if tracker.ckpt.step.numpy() > 0:
    print("Restored from step:", tracker.ckpt.step.numpy() + 1)
  else:
    print("Step:", tracker.ckpt.step.numpy() + 1)

  counter = 0
  loss, acc, auc = get_metrics()
  while tracker.ckpt.step < config["training"]["max_steps"]:
    for batch in data.batcher(mode='train'):
      counter += 1
      tokens, edges, label, _, _ = batch
      token_mask = tf.clip_by_value(tf.reduce_sum(tokens, -1), 0, 1)

      with tf.GradientTape() as tape:
        preds = model(tokens, token_mask, edges, training=True)
        ls, acs, aus = model.get_loss(preds, token_mask, label)
        update_metrics(loss, acc, auc, ls, acs, aus)

      grads = tape.gradient(ls, model.trainable_variables)
      grads, _ = tf.clip_by_global_norm(grads, 0.25)
      optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))

      samples = tf.shape(token_mask)[0]
      prev_samples = tracker.get_samples()
      curr_samples = tracker.update_samples(samples)
      if counter % config['training']['print_freq'] == 0:
        print("Progress: loss: {0:.3f}, acc: {1:.2%}, auc: {2:.2%}".format(ls, acs, aus))

      if counter % config['data']['valid_interval'] == 0:
        avg_accs = evaluate(data, config, model)
        tracker.save_checkpoint(model, avg_accs)
        print("Saving the best model...")

  print("Final Results: loss: {0:.3f}, acc: {1:.2%}, auc: {2:.2%}".format(loss.numpy(), acc.numpy(), auc.numpy()))

def evaluate(data, config, model, is_heldout=True):  # Similar to train, just without gradient updates
  loss, acc, auc = get_metrics()
  for batch in data.batcher(mode='dev' if is_heldout else 'eval'):
    tokens, edges, label, _, _ = batch
    token_mask = tf.clip_by_value(tf.reduce_sum(tokens, -1), 0, 1)
    preds = model(tokens, token_mask, edges, training=False)
    ls, acs, aus = model.get_loss(preds, token_mask, label)
    update_metrics(loss, acc, auc, ls, acs, aus)
  return acc.result().numpy(), auc.result().numpy()

def get_metrics():
  loss = tf.keras.metrics.Mean()
  acc = tf.keras.metrics.Mean()
  auc = tf.keras.metrics.Mean()
  return loss, acc, auc

def update_metrics(loss, acc, auc, ls, acs, aus):
  loss.update_state(ls)
  acc.update_state(acs)
  auc.update_state(aus)

if __name__ == '__main__':
  main()
