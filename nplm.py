# nplm -- a theano re-implementation of (Vaswani et al. 2013)
# To improve numerical stability, we used Adadelta (Zeiler 2012) as optimization method.

# August, 2016
# Shuoyang Ding @ Johns Hopkins University

import argparse
import logging
import os.path
import sys

import numpy as np
import theano
import theano.tensor as T

import lasagne as L
from loss import NCE
from utils import rand
from utils.misc import indexer
from utils.misc import numberizer

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

# constants
UNK = "<unk>"
BOS = "<s>"
EOS = "</s>"

parser = argparse.ArgumentParser()
parser.add_argument("--training-file", "-t", dest="training_file", metavar="PATH", help="File used as training corpus.", required=True)
parser.add_argument("--working-dir", "-w", dest="working_dir", metavar="PATH", help="Directory used to dump models etc.", required=True)
# parser.add_argument("--validation-file", dest="validation_file", metavar="PATH", help="Validation corpus used for stopping criteria.")
parser.add_argument("--decay-rate", dest="decay_rate", type=float, metavar="FLOAT", help="Decay rate as required by Adadelta (default = 0.95).")
parser.add_argument("--epsilon", "-e", dest="epsilon", type=float, metavar="FLOAT", help="Constant epsilon as required by Adadelta (default = 1e-6).")
parser.add_argument("--vocab-size", dest="vocab_size", type=int, metavar="INT", help="Vocabulary size of the language model (default = 500000).")
parser.add_argument("--word-dim", dest="word_dim", type=int, metavar="INT", help="Dimension of word embedding (default = 150).")
parser.add_argument("--hidden-dim1", dest="hidden_dim1", type=int, metavar="INT", help="Dimension of hidden layer 1 (default = 150).")
parser.add_argument("--hidden-dim2", dest="hidden_dim2", type=int, metavar="INT", help="Dimension of hidden layer 2 (default = 750).")
parser.add_argument("--noise-sample-size", "-k", dest="noise_sample_size", type=int, metavar="INT", help="Size of the noise sample per training instance for NCE (default = 100).")
parser.add_argument("--n-gram", "-n", dest="n_gram", type=int, metavar="INT", help="Size of the N-gram (default = 5).")
parser.add_argument("--max-epoch", dest="max_epoch", type=int, metavar="INT", help="Maximum number of epochs should be performed during training (default = 5).")
parser.add_argument("--batch-size", "-b", dest="batch_size", type=int, metavar="INT", help="Batch size (in sentences) of SGD (default = 1000).")
parser.add_argument("--save-interval", dest="save_interval", type=int, metavar="INT", help="Saving model only for every several updates (default = 100000).")

parser.set_defaults(
  decay_rate=0.95,
  epsilon=1e-6,
  vocab_size=500000,
  word_dim=150,
  hidden_dim1=150,
  hidden_dim2=750,
  noise_sample_size=100,
  n_gram=5,
  max_epoch=5,
  batch_size=1000,
  save_interval=1)

if theano.config.floatX=='float32':
  floatX = np.float32
else:
  floatX = np.float64

class nplm:

  # the default noise_distribution is uniform
  def __init__(self, n_gram, vocab_size, word_dim=150, hidden_dim1=150, hidden_dim2=750, noise_sample_size=100, batch_size=1000, decay_rate=0.95, epsilon=1e-6, noise_dist=[]):
    self.n_gram = n_gram
    self.vocab_size = vocab_size
    self.word_dim = word_dim
    self.hidden_dim1 = hidden_dim1
    self.hidden_dim2 = hidden_dim2
    self.noise_sample_size = noise_sample_size
    self.batch_size = batch_size
    self.decay_rate = decay_rate
    self.epsilon = epsilon
    
    # noise_dist must be shared to enable advanced indexing
    self.noise_dist = theano.shared(noise_dist, name='nd') \
        if noise_dist != [] \
        else theano.shared(np.array([floatX(1. / vocab_size)] * vocab_size, dtype=floatX), name = 'nd')
    self.D = theano.shared(
        np.random.uniform(-0.01, 0.01, (vocab_size, word_dim)).astype(floatX),
        name = 'D')
    self.C = theano.shared(
        np.random.uniform(-0.01, 0.01, (word_dim * n_gram, hidden_dim1)).astype(floatX),
        name = 'C')
    self.M = theano.shared(
        np.random.uniform(-0.01, 0.01, (hidden_dim1, hidden_dim2)).astype(floatX),
        name = 'M')
    self.E = theano.shared(
        np.random.uniform(-0.01, 0.01, (hidden_dim2, vocab_size)).astype(floatX),
        name = 'E')
    self.Cb = theano.shared(
        np.ones(hidden_dim1).astype(floatX) * -np.log(vocab_size).astype(floatX),
        name = 'Cb')
    self.Mb = theano.shared(
        np.ones(hidden_dim2).astype(floatX) * -np.log(vocab_size).astype(floatX),
        name = 'Mb')
    self.Eb = theano.shared(
        np.ones(vocab_size).astype(floatX) * -np.log(vocab_size).astype(floatX),
        name = 'Eb')

    self.__theano_init__()

  def __theano_init__(self):

    # Theano tensor for I/O 
    X = T.lmatrix('X')
    Y = T.lvector('Y')
    N = T.lvector('N')

    # network structure
    l_in = L.layers.InputLayer(shape=(self.batch_size, self.n_gram), input_var = X)
    l_we = L.layers.EmbeddingLayer(l_in, self.vocab_size, self.word_dim, W = self.D)
    l_f1 = L.layers.DenseLayer(l_we, self.hidden_dim1, W = self.C, b = self.Cb)
    l_f2 = L.layers.DenseLayer(l_f1, self.hidden_dim2, W = self.M, b = self.Mb)
    l_out = L.layers.DenseLayer(l_f2, self.vocab_size, W = self.E, b = self.Eb, nonlinearity=None)
    
    # lasagne.layers.get_output produces a variable for the output of the net
    O = L.layers.get_output(l_out) # (batch_size, vocab_size)

    lossfunc = NCE(self.batch_size, self.vocab_size, self.noise_dist, self.noise_sample_size)
    loss = lossfunc.evaluate(O, Y, N)
    # loss = T.nnet.categorical_crossentropy(O, Y).mean()

    # Retrieve all parameters from the network
    all_params = L.layers.get_all_params(l_out, trainable=True)

    # Compute AdaGrad updates for training
    updates = L.updates.adadelta(loss, all_params)

    # Theano functions for training and computing cost
    self.train = theano.function([l_in.input_var, Y, N], loss, updates=updates, allow_input_downcast=True)
    self.compute_loss = theano.function([l_in.input_var, Y, N], loss, allow_input_downcast=True)
    self.weights = theano.function(inputs = [], outputs = [self.D, self.C, self.M, self.E, self.Cb, self.Mb, self.Eb])

# ==================== END OF NPLM CLASS DEF ====================

def dump_matrix(m, model_file):
    np.savetxt(model_file, m, fmt="%.6f", delimiter='\t')

def dump(net, model_dir, options, vocab):
    model_file = open(model_dir, 'w')

    # config
    model_file.write("\\config\n")
    model_file.write("version 1\n")
    model_file.write("ngram_size {0}\n".format(options.n_gram + 1))
    model_file.write("input_vocab_size {0}\n".format(options.vocab_size))
    model_file.write("output_vocab_size {0}\n".format(options.vocab_size))
    model_file.write("input_embedding_dimension {0}\n".format(options.word_dim))
    model_file.write("num_hidden {0}\n".format(options.hidden_dim1))
    model_file.write("output_embedding_dimension {0}\n".format(options.hidden_dim2))
    model_file.write("activation_function rectifier\n\n") # currently only supporting rectifier... 

    # input_vocab
    model_file.write("\\input_vocab\n")
    for word in vocab:
      model_file.write(word + "\n")
    model_file.write("\n")
    model_file.write("\\output_vocab\n")
    for word in vocab:
      model_file.write(word + "\n")
    model_file.write("\n")

    [D, C, M, E, Cb, Mb, Eb] = net.weights()

    # input_embeddings
    model_file.write("\\input_embeddings\n")
    dump_matrix(D, model_file)
    model_file.write("\n")

    # hidden_weights 1
    model_file.write("\\hidden_weights 1\n")
    dump_matrix(np.transpose(C), model_file)
    model_file.write("\n")

    # hidden_biases 1
    model_file.write("\\hidden_biases 1\n")
    dump_matrix(Cb, model_file)
    model_file.write("\n")

    # hidden_weights 2
    model_file.write("\\hidden_weights 2\n")
    dump_matrix(np.transpose(M), model_file)
    model_file.write("\n")

    # hidden_biases 2
    model_file.write("\\hidden_biases 2\n")
    dump_matrix(Mb, model_file)
    model_file.write("\n")

    # output_weights
    model_file.write("\\output_weights\n")
    dump_matrix(np.transpose(E), model_file)
    model_file.write("\n")

    # output_biases
    model_file.write("\\output_biases\n")
    dump_matrix(Eb, model_file)
    model_file.write("\n")

    model_file.write("\\end")
    model_file.close()

def shuffle(indexed_ngrams, predictions):
  logging.info("shuffling data... ")
  arr = np.arange(len(indexed_ngrams))
  np.random.shuffle(arr)
  indexed_ngrams_shuffled = indexed_ngrams[arr, :]
  predictions_shuffled = predictions[arr]
  return (indexed_ngrams_shuffled, predictions_shuffled)

def sgd(examples, net, vocab, options, epoch, noise_dist):
  logging.info("epoch {0} started".format(epoch))  
  instance_count = 0
  batch_count = 0
  # for performance issue, if the remaining data is smaller than batch_size, we will just discard them
  X = []
  Y = []
  for example in examples:
    X.append(example[0])
    Y.append(example[1])
    instance_count += 1
    if instance_count % options.batch_size == 0:
      X = np.array(X)
      Y = np.array(Y)
      N = np.array(rand.distint(noise_dist, (options.noise_sample_size,)), dtype='int64') # (batch_size, noise_sample_size)
      net.train(X, Y, N)
      batch_count += 1
      X = []
      Y = []
      if batch_count % 1 == 0:
        logging.info("{0} instances seen".format(instance_count))
      if batch_count % options.save_interval == 0:
        logging.info("dumping models after {0} updates in epoch {1}".format(batch_count, epoch))
    	dump(net, options.working_dir + "/nplm.model.iter{0}.{1}".format(batch_count, epoch), options, vocab)
  # N = np.array(rand.distint(noise_dist, (len(indexed_ngrams), options.noise_sample_size)))
  # total_loss = net.compute_loss(indexed_ngrams, predictions)
  # logging.info("epoch {0} finished with NCE loss {1}".format(epoch, total_loss))
  logging.info("epoch {0} finished".format(epoch))

def create_lazy_examples(trnz, bos_index):
  len_trnz = len(trnz)
  for linen in rand.shuffled_xrange(0, len_trnz):
    numberized_line = trnz[linen]
    indexed_sentence = [bos_index] * (options.n_gram - 2)
    indexed_sentence.extend(numberized_line)
    for start in range(len(indexed_sentence) - options.n_gram):
      yield (indexed_sentence[start: start + options.n_gram], indexed_sentence[start + options.n_gram])

def main(options):

  options.n_gram -= 1

  # make training dir
  if not (os.path.isdir(options.working_dir) or os.path.exists(options.working_dir)):
    os.makedirs(options.working_dir)
  elif not os.path.isdir(options.working_dir):
    logging.fatal("cannot create training directory because a file already exists.")
    sys.exit(1)

  # collecting vocab
  logging.info("start collecting vocabulary")
  indexed_ngrams = []
  predictions = []
  nz = numberizer(limit = options.vocab_size, unk = UNK, bos = BOS, eos = EOS)
  (trnz, vocab, unigram_count) = nz.numberize(options.training_file)
  
  """
  for numberized_line in trnz:
    # think of a sentence with only 1 word w0 and we are extracting trigrams (n_gram = 3):
    # the numerized version would be "<s> w0 </s>".
    # after the sentence is augmented with 1 extra "<s>" at the beginning (now has length 4), 
    # we want to extract 1 trigram: [<s>, <s>, w0] (note that we don't want [<s>, w0, </s>])
    indexed_sentence = [bos_index] * (options.n_gram - 2)
    indexed_sentence.extend(numberized_line)
    for start in range(len(indexed_sentence) - options.n_gram):
      indexed_ngrams.append(indexed_sentence[start: start + options.n_gram])
      if start + options.n_gram < len(indexed_sentence):
        predictions.append(indexed_sentence[start + options.n_gram])
  del trnz
  """

  # build quick vocab indexer
  v2i = {}
  for i in xrange(len(vocab)):
    v2i[vocab[i]] = i

  total_unigram_count = floatX(sum(unigram_count.values()))
  unigram_dist = [floatX(0.0)] * len(unigram_count)
  # pdb.set_trace()
  for key in unigram_count.keys():
    unigram_dist[v2i[key]] = floatX(unigram_count[key] / total_unigram_count)
  del unigram_count
  unigram_dist = np.array(unigram_dist, dtype=floatX)
  logging.info("vocabulary collection finished")

  # training
  if len(vocab) < options.vocab_size:
    logging.warning("The actual vocabulary size of the training corpus {0} ".format(len(vocab)) + 
      "is smaller than the vocab_size option as specified {0}. ".format(options.vocab_size) + 
      "We don't know what will happen to nplm in that case, but for safety we'll decrease vocab_size as the vocabulary size in the corpus.")
    options.vocab_size = len(vocab)
  logging.info("start training with n-gram size {0}, vocab size {1}, decay_rate {2}, epsilon {3}, "
      .format(options.n_gram, len(vocab), options.decay_rate, options.epsilon) + 
      "word dimension {0}, hidden dimension 1 {1}, hidden dimension 2 {2}, noise sample size {3}"
      .format(options.word_dim, options.hidden_dim1, options.hidden_dim2, options.noise_sample_size))
  net = nplm(options.n_gram, len(vocab), options.word_dim, options.hidden_dim1, options.hidden_dim2,
      options.noise_sample_size, options.batch_size, options.decay_rate, options.epsilon, unigram_dist)
  bos_index = vocab.index(BOS)
  for epoch in range(1, options.max_epoch + 1):
    examples = create_lazy_examples(trnz, bos_index) 
    sgd(examples, net, vocab, options, epoch, unigram_dist)
    logging.info("dumping models after epoch {0}".format(epoch))
    dump(net, options.working_dir + "/nplm.model.{0}".format(epoch), options, vocab)
  logging.info("training finished")

if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
          parser.parse_known_args()[1]))
  main(options)

