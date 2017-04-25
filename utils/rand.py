# rand -- all the utilities that's related to random numbers
# 
# proudly developed by
# Shuoyang Ding @ Johns Hopkins University
# March, 2016

from bisect import bisect_left
import copy
import numpy as np
from numpy.random import random
from random import randint

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

lfsr_roots = [
    [2, 1],
    [3, 2],
    [4, 3],
    [5, 3],
    [6, 5],
    [7, 6],
    [8, 6, 5, 4],
    [9, 5],
    [10, 7],
    [11, 9],
    [12, 6, 4, 1],
    [13, 4, 3, 1],
    [14, 5, 3, 1],
    [15, 14],
    [16, 15, 13, 4],
    [17, 14],
    [18, 11],
    [19, 6, 2, 1],
    [20, 17],
    [21, 19],
    [22, 21],
    [23, 18],
    [24, 23, 22, 17],
    [25, 22],
    [26, 6, 2, 1],
    [27, 5, 2, 1],
    [28, 25],
    [29, 27],
    [30, 6, 4, 1],
    [31, 28],
    [32, 22, 2, 1],
    [33, 20],
    [34, 27, 2, 1],
    [35, 33],
    [36, 25],
    [37, 5, 4, 3, 2, 1],
    [38, 6, 5, 1],
    [39, 35],
    [40, 38, 21, 19],
    [41, 38],
    [42, 41, 20, 19],
    [43, 42, 38, 37],
    [44, 43, 18, 17],
    [45, 44, 42, 41],
    [46, 45, 26, 25],
    [47, 42],
    [48, 47, 21, 20],
]

# code copied from http://www.christopia.net/blog/lazy-shuffled-list-generator
# credit goes to Christopher J. MacLellan
def shuffled_xrange(start, stop=None, step=1):
    """
    This generates the full range and shuffles it using a Fibonacci linear
    feedback shift register:
        https://en.wikipedia.org/wiki/Linear_feedback_shift_register#Fibonacci_LFSRs
    Here I use a table of precomputed primitive roots of different polynomials
    mod 2. In many ways this is similar to the multiplicative congruential
    generator in that we are iterating through elements of a finite field. We
    need primitive roots so that we can be sure we generate all elements in the
    range.  If we get elements outside the range we ignore them and continue
    iterating. Finally, we need the generator to be a primitive root of the
    selected modulus, so that we generate a full cycle. The seed provides the
    randomness for the permutation.
    This function has the same args as the builtin ``range'' function, but
    returns the values in shuffled order:
        range(stop)
        range(start, stop[, step])
    >>> sorted([i for i in lazyshuffledrange3(10)])
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> sorted([i for i in lazyshuffledrange3(2, 20, 3)])
    [2, 5, 8, 11, 14, 17]
    """
    if stop is None:
        stop = start
        start = 0
    l = (stop - start) // step
    root_idx = l.bit_length()-2
    nbits = lfsr_roots[root_idx][0]
    roots = lfsr_roots[root_idx][1]
    nbits = l.bit_length()
    roots = lfsr_roots[nbits-2]
    seed = randint(1,l)
    lfsr = seed
    while (True):
        if lfsr <= l:
            yield step * (lfsr - 1) + start
        bit = 0
        for r in roots:
            bit = (bit ^ (lfsr >> (nbits - r)))
        bit = (bit & 1)
        lfsr =  (lfsr >> 1) | (bit << (nbits - 1))
        if (lfsr == seed):
            break

