#!/usr/bin/env python
# coding: utf-8

import numpy as np
from embedders import keras_embedder, embedding_datasets
from embedders.word2vec_embedder import Word2VecEmbedder
import gensim.downloader as api
from gensim.models import KeyedVectors
from correlation_test import *

# fixes weird OMP mac bug
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.random.seed(0)

embedder = None

gensim_model_name = 'glove-wiki-gigaword-50'
gensim_fn = f'./data/{gensim_model_name}.kv'

if not os.path.exists(gensim_fn):
    # we only want to do this once
    gensim_model = api.load(gensim_model_name)
    gensim_model.save(gensim_fn)

print('loading word2vec model...')
gensim_model = KeyedVectors.load(gensim_fn, mmap='r')  # mmap makes this significantly faster (lazy loading)
print('successfully loaded model')
embedder = Word2VecEmbedder(model=gensim_model)
dataset = embedder.dataset

forward_PC_names, _ = embedder.auto_name(forward=True, vectors=True)
correlation_test(embedder, forward_PC_names)
