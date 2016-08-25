# rand -- all the utilities that's related to random numbers
# 
# proudly developed by
# Shuoyang Ding @ Johns Hopkins University
# March, 2016

from bisect import bisect_left
import copy
import numpy as np
from numpy.random import random

# randomly generate a integer from a given distribution
# 
# dist is the discrete distribution, which is a list of float numbers
# for example, distribution of random variable x:
# 
# p(x = 0) = 0.5;
# p(x = 1) = 0.2;
# p(x = 2) = 0.3.
# 
# will be represented as [0.5, 0.2, 0.3]
# 
# the integer generated is in the range of [low, low + len(dist) - 1] (inclusive)
def distint(dist, size, low = 0):
  cumulative_dist = accumulate_dist(dist)
  return cum_distint(cumulative_dist, size, low)

# same as distint but reads a cumulative distribution
# if you needs to draw a lot of samples, this might be slightly more efficient... 
def cum_distint(cumdist, size, low = 0):
  sample_size = np.prod(size)
  samples = random((sample_size,))
  ints = []
  for sample in samples:
    i = bisect_left(cumdist, sample)
    if i != len(cumdist):
      ints.append(i + low)
    else:
      raise Exception("bisect failed with query {0} into {1}: ill-formed distribution?".format(sample, cumdist))
  return np.array(ints).reshape(size)

# transform distribution density into a cumulative distribution
def accumulate_dist(dist):
  # transform to cumulative distribution
  cumulative_dist = copy.copy(dist)
  for i in range(1, len(cumulative_dist)):
    cumulative_dist[i] = cumulative_dist[i - 1] + dist[i]

  # check for normalization
  if abs(1.0 - cumulative_dist[-1]) > 1e-3:
    raise Exception("distribution {0} not normalized!".format(dist))
  # help with float point precision 
  # i.e. when not normalized, force it to normalize
  else:
    for i in range(-1, -len(cumulative_dist), -1):
      if cumulative_dist[i] > 1.0:
        cumulative_dist[i] = 1.0
      else:
        cumulative_dist[i] = 1.0
        break

  return cumulative_dist

