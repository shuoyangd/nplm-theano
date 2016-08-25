# rnlm -- a theano-based recurrent neural network language model
# 
# proudly developed by
# Shuoyang Ding @ Johns Hopkins University
# 
# with the help from this blog post:
# http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
# 
# March, 2016

import argparse
from indexer import indexer
import logging
import numpy as np
import pdb
import pickle
from rnn import rnn, rnn_batch
import sys

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--training-file", "-t", dest="training_file", metavar="PATH", help="File used as training corpus.", required=True)
parser.add_argument("--working-dir", "-w", dest="working_dir", metavar="PATH", help="Directory used to dump models etc.", required=True)
parser.add_argument("--learning-rate", dest="learning_rate", type=float, metavar="FLOAT", help="Learning rate used to update weights (default = 0.25).")
parser.add_argument("--hidden-dim", dest="hidden_dim", type=int, metavar="INT", help="Dimension of the hidden layer (default = 100).")
parser.add_argument("--bptt-truncate", dest="bptt_truncate", type=int, metavar="INT", help="Maximum bptt level (default = -1). Pass -1 if you don't want to truncate.")
parser.add_argument("--batch-size", "-b", dest="batch_size", type=int, metavar="INT", help="Batch size (in sentences) of SGD (default = 1).")
parser.add_argument("--gradient-check", dest="gradient_check", type=int, metavar="INT", help="The iteration interval for gradient check. Pass 0 if gradient check should not be performed (default = 0).")
parser.add_argument("--gradient-clipping", dest="gradient_clipping", type=float, metavar="FLOAT", help="The threshold used for gradient clipping of the training gradient (default = inf).")
parser.add_argument("--save-interval", dest="save_interval", type=int, metavar="INT", help="The epoch interval for saving models. Pass 0 if wish to save only once at the end of each epoch (default = 0).")
parser.add_argument("--max-epoch", dest="max_epoch", type=int, metavar="INT", help="Maximum number of epochs should be performed during training (default = 5).")

parser.set_defaults(
  learning_rate=0.25,
  hidden_dim=100,
  bptt_truncate=-1,
  batch_size=1,
  gradient_check=0,
  gradient_clipping=np.inf,
  save_interval=0,
  max_epoch=5)

def sgd(indexed_corpus, predictions, net, options, epoch):
  logging.info("epoch {0} started".format(epoch))
  instance_count = 0
  for (x, y) in zip(indexed_corpus, predictions):
    net.sgd(x, y, options.learning_rate)
    instance_count += 1
    if instance_count % 1 == 0:
      logging.info("{0} instances seen".format(instance_count))
    if options.gradient_check != 0 and instance_count % options.gradient_check == 0:
      net.gradient_check(x, y)
    if options.save_interval != 0 and instance_count % options.save_interval:
      # TODO: supposed to save model here
      pass
  total_loss = net.total_loss(indexed_corpus, predictions)
  logging.info("epoch {0} finished with loss {1}".format(epoch, total_loss))

def sgd_batch(indexed_corpus, predictions, net, options, epoch):
  logging.info("epoch {0} started".format(epoch))
  instance_count = 0
  batch_count = 0
  for start in range(0, len(indexed_corpus), options.batch_size):
    X = indexed_corpus[start: min(start + options.batch_size, len(indexed_corpus))]
    Y = predictions[start: min(start + options.batch_size, len(indexed_corpus))]
    net.sgd(X, Y, options.learning_rate)
    instance_count += min(options.batch_size, len(indexed_corpus) - start)
    batch_count += 1
    if batch_count % 100 == 0:
      logging.info("{0} instances seen".format(instance_count))
    if options.gradient_check != 0 and batch_count % options.gradient_check == 0:
      net.gradient_check(x, y)
    if options.save_interval != 0 and batch_count % options.save_interval:
      # supposed to save model here
      pass
  total_loss = net.total_loss(indexed_corpus, predictions)
  logging.info("epoch {0} finished with loss {1}".format(epoch, total_loss))

def main(options):
  # collecting vocab
  logging.info("start collecting vocabulary")
  training_corpus = open(options.training_file)
  indexed_corpus = []
  vocab = indexer()
  vocab.add("</s>")
  for sentence in training_corpus:
    tokens = ["<s>"]
    tokens.extend(sentence.strip().split(' '))
    indexed_sentence = []
    for token in tokens:
      ix = vocab.getIndex(token)
      indexed_sentence.append(ix)
    indexed_corpus.append(indexed_sentence)
  logging.info("vocabulary collection finished")
  training_corpus.close()
  predictions = [x[1:] for x in indexed_corpus]
  map(lambda y: y.append(vocab.indexOf("</s>")), predictions)

  # training
  logging.info("start training with vocabulary size {0}, learning rate {1}, hidden dimension {2}, bptt truncate {3}, clipping threshold {4}"
      .format(vocab.size(), options.learning_rate, options.hidden_dim, options.bptt_truncate, options.gradient_clipping))
  if options.batch_size == 1:
    net = rnn(vocab.size(), options.hidden_dim, options.bptt_truncate, options.gradient_clipping)
    for epoch in range(options.max_epoch):
      sgd(indexed_corpus, predictions, net, options, epoch)
  else:
    net = rnn_batch(vocab.size(), options.hidden_dim, options.bptt_truncate, options.gradient_clipping,
        options.batch_size, vocab.getIndex("</s>"))
    for epoch in range(options.max_epoch):
      sgd_batch(indexed_corpus, predictions, net, options, epoch)
  logging.info("trainining finished")
 
if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
          parser.parse_known_args()[1]))
  main(options)

