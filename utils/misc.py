from collections import Counter
import logging
import pickle
import sys

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

# a general indexer of objects
# but designed with word indexer of a language model in mind
# 
class indexer:
  def __init__(self):
    self.objects = []
    self.indexes = {}
    self.locked = False

  def getIndex(self, e):
    if e in self.indexes:
      return self.indexes[e]
    else:
      ix = len(self.objects)
      self.indexes[e] = ix
      self.objects.append(e)
      return ix

  def indexOf(self, e):
    if e in self.indexes:
      return self.indexes[e]
    else:
      return -1

  def add(self, e):
    if self.locked == True:
      raise Exception("attempt to add element into a locked indexer")
    if e in self.indexes:
      return False
    else:
      self.indexes[e] = len(self.objects)
      self.objects.append(e)
      return True

  def size(self):
    return len(self.objects)

  def lock(self):
    self.locked = True

  def getObjects(self):
    return self.objects

# numberizer -- numberize a tokenized text corpora and optionally truncate the vocabulary
# (substitute the rest with a special symbol)
# designed with language model application in mind
#
class numberizer:

  # vocabulary limit = 0 means no vocabulary truncating will be performed
  # setting either start or end to "None" will stop the numberizer
  #   from adding start and ending symbol to the sentence
  # 
  # the reason to set these symbols is to avoid undesirable collisions with tokens
  #   that are actually part of the corpus
  def __init__(self, limit=0, unk="<unk>", bos="<s>", eos="</s>"):
    self.limit = limit
    self.unk = unk
    self.bos = bos
    self.eos = eos
    self.dont_augment_bos_eos = not bos or not eos

  # the three returned values are:
  # + numberized corpus
  # + a list of vocabulary: you can use it as an indexer -- 
  #     it's guaranteed to provide the same index as the numberized corpus
  # + a counter of the raw tokens (not numberized but truncated) 
  def numberize(self, text_dir, numberized_dir = None):
    # first scan: collect
    text_file = open(text_dir)
    cnt = Counter()
    if self.limit != 0:
        vocab = [self.unk]
    else:
        vocab = []
    linen = 1
    logging.info("Starting first scan of the training corpus.")
    for line in text_file:
      if linen % 100000 == 0:
        logging.info("{0} lines scanned.".format(linen))
      if not self.dont_augment_bos_eos:
        cnt[self.bos] += 1
        cnt[self.eos] += 1
      tokens = line.strip().split(' ')
      for token in tokens:
        cnt[token] += 1
      linen += 1
    if self.limit != 0:
      pairs = cnt.most_common(self.limit - 1) # leave a space for <unk>
      vocab.extend([pair[0] for pair in pairs])
    else:
      vocab.extend(list(cnt.elements()))
    text_file.close()
    logging.info("First scan of the training corpus finished.")

    # build fast indexer
    vocab_indexer = {}
    for i in range(len(vocab)):
      vocab_indexer[vocab[i]] = i

    # remove stop words from counter and add their counts to unk
    if self.limit != 0:
      cnt[self.unk] = 0 # should have unk in counter anyway to keep length consistent
      for key in cnt.keys():
        if not key in vocab_indexer:
          oov_count = cnt[key]
          del cnt[key]
          cnt[self.unk] += oov_count
    
    # second scan: numberize and truncate
    text_file = open(text_dir)
    numberized = []
    unk_index = vocab_indexer[self.unk]
    bos_index = vocab_indexer[self.bos]
    eos_index = vocab_indexer[self.eos]
    linen = 1
    logging.info("Starting second scan of the training corpus.")
    for line in text_file:
      if linen % 100000 == 0:
        logging.info("{0} lines scanned.".format(linen))
      numberized_line = []
      if not self.dont_augment_bos_eos:
        numberized_line.append(bos_index)
      tokens = line.strip().split(' ')
      for token in tokens:
        if self.limit == 0: # if vocab truncating is not imposed, don't bother
          numberized_line.append(vocab_indexer[token])
        elif token in vocab_indexer: # in-vocabulary
          numberized_line.append(vocab_indexer[token])
        else: # OOV
          numberized_line.append(unk_index)
      if not self.dont_augment_bos_eos:
        numberized_line.append(eos_index)
      numberized.append(numberized_line)
      linen += 1
    text_file.close()
    del vocab_indexer
    logging.info("Second scan of the training corpus finished.")

    # dump
    if numberized_dir:
      numberized_file = open(numberized_dir)
      pickle.dump(numberized, numberized_file)
      numberized_file.close()

    return (numberized, vocab, cnt)

