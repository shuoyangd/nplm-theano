import numpy as np
import theano.tensor as T

class NCE:
  def __init__(self, batch_size, vocab_size, noise_dist, noise_sample_size):
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.noise_dist = noise_dist
    self.noise_sample_size = noise_sample_size

  # O is the output of the network, 
  #   should have shape of (batch_size, vocab_size)
  # N is the generated noise sample (each element is an integer vocabulary index),
  #   should have shape of (noise_sample_size,)
  def evaluate(self, O, Y, N):
    
    # w is the next word in the training data
    pw = O[np.arange(0, self.batch_size), Y]
    qw = self.noise_dist[Y]
    # wb is the noise word in the noise samples
    pwb = T.take(O, N) # (noise_sample_size, )
    qwb = T.take(self.noise_dist, N) # (noise_sample_size, )
    
    # P(D = 1 | c, w)
    pd1 = pw / (pw + self.noise_sample_size * qw) # (batch_size, )
    # P(D = 0 | c, wb)
    pd0 = (self.noise_sample_size * qwb) / (pwb + self.noise_sample_size * qwb) # (noise_sample_size, )

    return T.sum(T.log(pd1) + T.sum(T.log(pd0))) # scalar

