# rnlm -- a theano-based recurrent neural network language model
# 
# proudly developed by
# Shuoyang Ding @ Johns Hopkins University
# 
# with the help from this blog post:
# http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
# 
# March, 2016

import logging
import numpy as np
import pdb
import theano
import theano.tensor as T
# from theano.compile.nanguardmode import NanGuardMode

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

# **********************************************************
# Vanilla rnn implementation
class rnn:

  def __init__(self, word_dim, hidden_dim=100, bptt_truncate=-1, clipping_threshold=np.inf):
    # Assign instance variables
    self.word_dim = word_dim
    self.hidden_dim = hidden_dim
    self.bptt_truncate = bptt_truncate
    self.clipping_threshold = clipping_threshold
    # Randomly initialize the network parameters
    self.U = theano.shared(
        np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim)),
        name = 'U')
    self.V = theano.shared(
        np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim)),
        name = 'V')
    self.W = theano.shared(
        np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim)),
        name = 'W')

    self.__theano_init__()
  
  def __theano_init__(self):
    x = T.ivector('x')
    y = T.ivector('y')
    
    def fprop_step(x_t, s_tm1, U, V, W):
      s = T.tanh(U[:, x_t] + W.dot(s_tm1))
      # softmax want a 2D tensor as input and spits out a 2d tensor output
      # that's why we only took o[0] as the return value
      o = T.nnet.softmax(V.dot(s))
      return [o[0], s]

    # the parameters passed to fn is in the order of :
    # sequence, outputs_info, non_sequences
    ([o, s], _) = theano.scan(fn=fprop_step,
        sequences = x,
        outputs_info = [None, dict(initial=T.zeros(self.hidden_dim))],
        non_sequences = [self.U, self.V, self.W],
        truncate_gradient=self.bptt_truncate,
        strict=True)

    prediction = T.argmax(o, axis=1)
    loss = T.sum(T.nnet.categorical_crossentropy(o, y))

    dU = T.grad(loss, self.U).clip(-self.clipping_threshold, self.clipping_threshold)
    dW = T.grad(loss, self.W).clip(-self.clipping_threshold, self.clipping_threshold)
    dV = T.grad(loss, self.V).clip(-self.clipping_threshold, self.clipping_threshold)

    lr = T.scalar('lr')

    # self.fprop = theano.function(inputs = [x], outputs = o)
    self.pred = theano.function(inputs = [x], outputs = prediction)
    self.loss = theano.function(inputs = [x, y], outputs = loss)
    self.bptt = theano.function(inputs = [x, y], outputs = [dU, dW, dV])
    self.sgd = theano.function([x, y, lr], [],
        updates = [(self.U, self.U - lr * dU), (self.W, self.W - lr * dW), (self.V, self.V - lr * dV)])
    
  def total_loss(self, X, Y):
    ret = 0.0
    for (x, y) in zip(X, Y):
      ret += self.loss(x, y)
    return ret

  def gradient_check(self, x, y, h=1e-6, threshold=1e-3):
    logging.info("performing gradient check... ")
    [dU, dW, dV] = self.bptt(x, y)
    loss = self.loss(x, y)
    deriv = {'U': dU, 'V': dV, 'W': dW}
    for (name, param_T) in zip(['U', 'V', 'W'], [self.U, self.V, self.W]):
      logging.info("checking d{0}... ".format(name))
      param = param_T.get_value()
      it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
      while not it.finished:
        ix = it.multi_index
        origval = param[ix]
        param[ix] = origval + h
        param_T.set_value(param)
        loss_p = self.loss(x, y)
        param[ix] = origval - h
        param_T.set_value(param)
        loss_m = self.loss(x, y)
        slope = (loss_p - loss_m) / (2 * h)
        param[ix] = origval
        param_T.set_value(param)
        # if deriv[name][ix] != 0.0 and abs(slope - deriv[name][ix]) > threshold:
        if abs(slope - deriv[name][ix]) > threshold:
          logging.warning("position {0} of the parameter {1} failed gradient check: \n"
              .format(str(ix), name) + 
              "slope value: " + str(slope) + '\n' +
              "gradient value: " + str(deriv[name][ix]) + '\n' + 
              "loss: " + str(loss))
          return 
        # elif deriv[name][ix] != 0.0:
          # logging.warning("non-zero position {0} passed gradient check".format(str(ix)))
        it.iternext()
    logging.info("gradient checking passed")



# **********************************************************
# RNN implementation with GPU-optimized batch update support
class rnn_batch(rnn):

  # padding_int is the integer used to form the sentence batch into a matrix
  def __init__(self, word_dim, hidden_dim=100, bptt_truncate=-1, clipping_threshold=np.inf, batch_size = 50, padding_int=0):
    # Assign instance variables
    self.word_dim = word_dim
    self.hidden_dim = hidden_dim
    self.bptt_truncate = bptt_truncate
    self.clipping_threshold = clipping_threshold
    self.batch_size = batch_size
    self.padding_int = padding_int
    # Randomly initialize the network parameters
    self.U = theano.shared(
        np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim)),
        name = 'U')
    self.V = theano.shared(
        np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim)),
        name = 'V')
    self.W = theano.shared(
        np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim)),
        name = 'W')

    self.__theano_init__()
  
  def __theano_init__(self):
    X = T.imatrix('X')
    Y = T.imatrix('Y')

    # note that here x_t is a vector (t-th token in all the sentences)
    # whereas S_tm1 is a matrix (each line is the hidden state corresponding to one of the sentences)
    def fprop_step(x_t, S_tm1, U, V, W):
      # U (word_dim, hidden_dim)
      # x_t (batch_size, )
      # W (hidden_dim, hidden_dim) 
      # S_tm1 (batch_size, hidden_dim)
      pdb.set_trace()
      Ux = U.take(x_t, axis=1).T # (batch_size, hidden_dim)
      (Wh, _) = theano.scan(fn = lambda s, W: W.dot(s),
          sequences = [S_tm1],
          outputs_info = None,
          non_sequences = W) # (batch_size, hidden_dim) #FIXME: actully (hidden_dim, hidden_dim)?
      S = T.tanh(Ux + Wh)
      (O, _) = theano.scan(fn = lambda s, V: V.dot(s),
          sequences = [S],
          outputs_info = None,
          non_sequences = V) # (batch_size, word_dim)
      return [O, S]

    # the parameters passed to fn is in the order of :
    # sequence, outputs_info, non_sequences
    ([O, S], _) = theano.scan(fn=fprop_step,
        sequences = X,
        outputs_info = [None, dict(initial=T.zeros((self.batch_size, self.hidden_dim)))],
        non_sequences = [self.U, self.V, self.W],
        truncate_gradient=self.bptt_truncate,
        strict=True)

    prediction = T.argmax(O, axis=1)
    # pdb.set_trace()
    (loss, _) = theano.scan(fn = lambda o, y: T.nnet.categorical_crossentropy(o, y),
        sequences = [O, Y],
        outputs_info = None,
        non_sequences = [])
    loss = loss.sum()

    dUb = T.grad(loss, self.U).clip(-self.clipping_threshold, self.clipping_threshold) # same shape as U
    dVb = T.grad(loss, self.V).clip(-self.clipping_threshold, self.clipping_threshold) # same shape as V
    dWb = T.grad(loss, self.W).clip(-self.clipping_threshold, self.clipping_threshold) # same shape as W

    lr = T.scalar('lr')

    # self.fprop = theano.function(inputs = [x], outputs = o)
    self.pred = theano.function(inputs = [X], outputs = prediction)
    self.loss = theano.function(inputs = [X, Y], outputs = loss)
    self.bptt = theano.function(inputs = [X, Y], outputs = [dUb, dWb, dVb])
    self.sgd_helper = theano.function([X, Y, lr], [],
        updates = [(self.U, self.U - lr * dUb),
            (self.V, self.V - lr * dVb),
            (self.W, self.W - lr * dWb)])

  # automatic padding is done here
  def sgd(self, X, Y, lr):
    # determine the longest sentence
    padded_length = 0
    for x in X:
      if padded_length < len(x):
        padded_length = len(x)
    
    # do the padding
    for (x, y) in zip(X, Y):
      pad = [self.padding_int] * (padded_length - len(x))
      x.extend(pad)
      y.extend(pad)

    self.sgd_helper(X, Y, lr)

