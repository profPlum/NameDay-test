#!/usr/bin/env python
# coding: utf-8

import numpy as np
from embedders import keras_embedder, embedding_datasets
from embedders.word2vec_embedder import Word2VecEmbedder
import gensim.downloader as api
from gensim.models import KeyedVectors

# fixes weird OMP mac bug
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.random.seed(0)

use_word2vec = True
retrain = True
embedder = None

if not use_word2vec:
    embedding_dims = 30  # aka num PCs
    num_bytes = 5000 * 5  # to process from text
    dataset = embedding_datasets.CorpusEmbeddingDataset("data/shakespear.txt", num_bytes)
    dataset.skip_gram_encoding(n=10)
    print(dataset.decode()[:10])

    embedder = keras_embedder.KerasEmbedder(dataset=dataset)

    if retrain:
        embedder.fit(epochs=10, embedding_dims=embedding_dims)
        embedder.save('./data/keras_embedder.h5')
    else:
        embedder.load('./data/keras_embedder.h5')
else:
    gensim_model_name = 'glove-twitter-25'
    gensim_fn = f'./data/{gensim_model_name}.kv'

    if not os.path.exists(gensim_fn):
        # we only want to do this once
        gensim_model = api.load(gensim_model_name)  # api.load("glove-wiki-gigaword-100")
        gensim_model.save(gensim_fn)

    print('loading word2vec model...')
    gensim_model = KeyedVectors.load(gensim_fn, mmap='r')  # mmap makes this significantly faster (lazy loading)
    print('successfully loaded model')
    gensim_embedder = Word2VecEmbedder(model=gensim_model)
    dataset = embedder.dataset

# from tests.correlation_test import *
# if not use_word2vec:  # too slow for word2vec
#     assert(np.all(embedder.soft_inverse(embedder.vocab_embeddings)[0] == embedder.dataset.vocab))
#     test_input = np.random.normal(size=(len(dataset.vocab), 1))
#     assert(np.all(np.isclose(embedder.embed(one_hots=test_input.T), test_input.T.dot(embedder.vocab_embeddings), atol=1e-5)))
#
# # print("group similarity score:", embedder.test_group_similarity())
#
# forward_PC_names, _ = embedder.auto_name(forward=True, vectors=True)
# backward_PC_names, _ = embedder.auto_name_from_similarity(normalize=True, vectors=True)
# compare_PC_name_algs(embedder, forward_PC_names, backward_PC_names,
#                      correlation_threshold=0.1)

from tests.part_whole_test import DirectPartWholeTester

part_whole_tester = DirectPartWholeTester(embedder)
part_whole_tester.load_part_whole_dataset('./data/part_whole_big.csv')

part_whole_tester.do_parts_from_whole_test()
print('done')

##################  Driver for explainable network  ##########################

# layer_sizes = [50, 50, 37, 25]
# explanation_index = 1  # index of layer to explain
# model = network_helpers.build_seq_model(layer_sizes)
#
# explainable_network = ExplainableNetwork(model, train=True)
# explainable_network.fit()
# explanation = explainable_network.explain(explanation_index, vocab_embeddings[:layer_sizes[-1]])
# print(f"hidden layer {explanation_index} neuron names:", embedder.soft_inverse(explanation))

# sys.exit(0)

##############################################################################
