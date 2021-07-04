import sys
sys.path.append('.')

import argparse
import yaml

import numpy as np
import tensorflow as tf

from checkpoint_tracker import Tracker
from data import data_loader, vocabulary
from meta_model import DefuseProgModel

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
  data = data_loader.DataLoader(args.data_path, config["data"], vocabulary.Vocabulary(args.vocabulary_path))
  if args.eval_only:
    if args.models is None or args.log is None:
      raise ValueError("Must provide a path to pre-trained models when running final evaluation")
    test(data, config, args.models, args.log)
  else:
    train(data, config, args.models, args.log)

def test(data, config, model_path, log_path):
  model = DefuseProgModel(config['model'], data.vocabulary.vocab_dim)
  model.run_dummy_input()
  tracker = Tracker(model, model_path, log_path)
  tracker.restore(best_model=True)
  evaluate(data, config, model, is_heldout=False)

def train(data, config, model_path=None, log_path=None):
  model = DefuseProgModel(config['model'], data.vocabulary.vocab_dim)
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

  mbs = 0
  loss, acc, auc = get_metrics()
  while tracker.ckpt.step < config["training"]["max_steps"]:
    for batch in data.batcher(mode='train'):
      mbs += 1
      tokens, edges, label, items1, items2 = batch
      token_mask = tf.clip_by_value(tf.reduce_sum(tokens, -1), 0, 1)

      with tf.GradientTape() as tape:
        pointer_preds = model(tokens, token_mask, edges, training=True)
        ls, acs, aus = model.get_loss(pointer_preds, token_mask, label, items1, items2)

      grads = tape.gradient(ls, model.trainable_variables)
      grads, _ = tf.clip_by_global_norm(grads, 0.25)
      optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))

      # Update statistics
      num_buggy = tf.reduce_sum(tf.clip_by_value(label, 0, 1))
      samples = tf.shape(token_mask)[0]
      prev_samples = tracker.get_samples()
      curr_samples = tracker.update_samples(samples)
      update_metrics(loss, acc, auc, token_mask, ls, acs, aus, num_buggy)

      # Every few minibatches, print the recent training performance
      if mbs % config["training"]["print_freq"] == 0:
        avg_losses = "{0:.3f}".format(loss.result().numpy())
        avg_accs = "{0:.2%}".format(acc.result().numpy())
        avg_aucs = "{0:.2%}".format(auc.result().numpy())
        print("MB: {0}, seqs: {1:,}, loss: {2}, accs: {3}, aucs: {4}".format(mbs, curr_samples, avg_losses, avg_accs, avg_aucs))
        loss.reset_states()
        acc.reset_states()
        auc.reset_states()

      # Every valid_interval samples, run an evaluation pass and store the most recent model with its heldout accuracy
      if prev_samples // config["data"]["valid_interval"] < curr_samples // config["data"]["valid_interval"]:
        avg_accs = evaluate(data, config, model)
        tracker.save_checkpoint(model, avg_accs)
        if tracker.ckpt.step >= config["training"]["max_steps"]:
          break
        else:
          print("Step:", tracker.ckpt.step.numpy() + 1)

def evaluate(data, config, model, is_heldout=True):  # Similar to train, just without gradient updates
  if is_heldout:
    print("Running evaluation pass on heldout data")
  else:
    print("Testing pre-trained model on full eval data")

  loss, acc, auc = get_metrics()
  mbs = 0
  for batch in data.batcher(mode='dev' if is_heldout else 'eval'):
    mbs += 1
    tokens, edges, label, items1, items2 = batch    
    token_mask = tf.clip_by_value(tf.reduce_sum(tokens, -1), 0, 1)

    pointer_preds = model(tokens, token_mask, edges, training=False)
    ls, acs, aus = model.get_loss(pointer_preds, token_mask, label, items1, items2)
    num_buggy = tf.reduce_sum(tf.clip_by_value(label, 0, 1))
    update_metrics(loss, acc, auc, token_mask, ls, acs, aus, num_buggy)
    if is_heldout and counts[0].result() > config['data']['max_valid_samples']:
      break
    if not is_heldout and mbs % config["training"]["print_freq"] == 0:
      avg_losses = "{0:.3f}".format(loss.result().numpy())
      avg_accs = "{0:.2%}".format(acc.result().numpy())
      avg_aucs = "{0:.2%}".format(auc.result().numpy())
      print("Testing progress: loss: {0}, accs: {1}, aucs: {2}".format(avg_losses, avg_accs, avg_aucs))

  avg_accs_str = "{0:.2%}".format(acc.result().numpy())
  avg_loss_str = "{0:.3f}".format(loss.result().numpy())
  avg_aucs_str = "{0:.2%}".format(aucs.result().numpy())
  print("Evaluation result: loss: {0}, accs: {1}, aucs: {2}".format(avg_loss_str, avg_accs_str, avg_aucs_str))
  return avg_accs

def get_metrics():
  loss = tf.keras.metrics.Mean()
  acc = tf.keras.metrics.Mean()
  auc = tf.keras.metrics.Mean()
  return loss, acc, auc

def update_metrics(loss, acc, auc, token_mask, ls, acs, aus, num_buggy_samples):
  num_samples = tf.shape(token_mask)[0]
  loss.update_state(ls)
  acc.update_state(acs)
  auc.update_state(aus)

if __name__ == '__main__':
  main()
