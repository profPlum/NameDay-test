import numpy as np
import network_helpers
import keras.layers as L

from gensim.models import KeyedVectors
from embedders import embedding_datasets
from embedders.embedder import Embedder


class Word2VecEmbedder(Embedder):
    def __init__(self, model_path=None, model=None):
        super().__init__(model_path)
        if model: self._embedder_model = model

        self._dataset = embedding_datasets.VocabEmbeddingDataset(gensim_model=self._embedder_model)
        if self._embedder_model: self._vocab_embeddings = self.embed(self._dataset.vocab)

    def _embed(self, words=None, one_hots=None):
        if one_hots is not None: words = self._dataset.decode(one_hots)
        embeddings = self._embedder_model[words]
        return embeddings

    def load(self, model_path):
        # Load pretrained model (since intermediate data is not included,
        # the model cannot be refined with additional data)
        self._embedder_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        self._vocab_embeddings = self.embed(self._dataset.vocab)
    
    @property
    def _embedding_weights(self):
        # NOTE: encode(vocab) = I because it is a one_hot of every possible vocab word
        # vocab_embeddings = W*I + b (s.t. b=0), so vocab_embeddings = W (assuming they are ordered which they are)
        return self._vocab_embeddings.T, 0
