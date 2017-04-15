# nplm-theano

nplm-theano is a re-implementation of the Neural Probablistic Language Model tookit ([nplm](http://nlg.isi.edu/software/nplm/)) by USC-ISI.

+ It has the same model dumping format as the nplm toolkit, so you can integrate it with other softwares such as [Moses](https://github.com/moses-smt/mosesdecoder).
+ The implementation is based on theano, so you are free to run it on GPU and speed up training.

### Dependencies

+ theano >= 0.8.0
+ lasagna >= 0.2.0dev1
+ PyTables >= 3.4.1

To use GPU for language model training, you may need the following optional packages as well:

+ CUDA >= 7
+ cuDNN >= 3 (to enable further theano optimization)

### Usage

All the argument usages can be printed by running:

`python nplm.py --help`

### Publications

+ Vaswani et al. 2013, Decoding with Large-Scale Neural Language Models Improves Translation: http://www.aclweb.org/anthology/D13-1140
+ Dyer 2014, Notes on Noise Contrastive Estimation and Negative Sampling: http://arxiv.org/abs/1410.8251
+ Zoph et al. 2016, Simple, Fast Noise-Contrastive Estimation for Large RNN Vocabularies: http://www.aclweb.org/anthology/N16-1145

