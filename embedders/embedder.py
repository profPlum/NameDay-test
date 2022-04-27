import keras
from embedders import embedding_datasets
import numpy as np
import scipy.stats as stats
import warnings
import network_helpers
from copy import copy
from copy import deepcopy
import scipy.stats


class Embedder:
    """ abstract embedder class that can use a number of backends """
    def __init__(self, model_path=None, dataset: embedding_datasets.VocabEmbeddingDataset = None):
        self._embedder_model = None  # initialize me
        self._dataset = dataset  # initialize me
        self._vocab_embeddings = None  # initialize me
        if model_path: self.load(model_path)

        self._original_vocab = None

    def load(self, model_path):
        raise NotImplementedError()

    def embed(self, words=None, one_hots=None):
        """ returns normalized embeddings (this effectively has us working in polar coordinates) """
        embeddings = self._embed(words=words, one_hots=one_hots)
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape([1, -1])
            # make into a matrix
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def _embed(self, words=None, one_hots=None):
        raise NotImplementedError()

    @property
    def _embedding_weights(self):
        """:return: W,b s.t. Wx+b=z"""
        raise NotImplementedError()

    def restrict_vocab(self, whitelist_vocab):
        self.reset_vocab()
        self._original_vocab = (self.dataset.vocab, self.vocab_embeddings)

        whitelist_mask = np.isin(self.dataset.vocab, set(whitelist_vocab))
        self._vocab_embeddings = self._vocab_embeddings[whitelist_mask]
        self._dataset._vocab = np.array(self.dataset.vocab)[whitelist_mask]

    def reset_vocab(self):
        if self._original_vocab is not None:
            self.dataset._vocab, self._vocab_embeddings = self._original_vocab
            self._original_vocab = None

    def inverse(self, embeddings):
        """
        takes the direct vocab-vector inverse of the embeddings
        :param embeddings: n x PC  embeddings
        """
        weights = self._embedding_weights

        # in a normal model layer output would be transpose of format we accept
        return network_helpers.invert_linear_layer(embeddings.T, W=weights[0], b=weights[1])

    def soft_inverse(self, embeddings, acceptable_names=lambda names: True, min_abstractness=None):
        """
        matches embedding to nearest vocab embedding then takes the inverse (directly to words)
        :param embeddings: n x PC  embeddings
        :param acceptable_names: predicate determining if a name is acceptable
        :param min_abstractness: if specified then predicate automatically requires this much abstractness for a name
        :return: words, cosine_similarity
        """

        if min_abstractness is not None:
            # in reality this is done one name at a time, though it could be more efficient if done at once
            acceptable_names = lambda names: self.get_abstractness_scores(names) > min_abstractness

        # this does exactly what we want but requires self.vocab_embeddings
        return self._dataset.soft_decode(embeddings, _target_mat=self.vocab_embeddings,
                                         acceptable_name=acceptable_names)

    @property
    def embedder_model(self):
        return self._embedder_model

    @property
    def dataset(self):
        return self._dataset

    @property
    def vocab_embeddings(self):
        return self._vocab_embeddings

    def auto_name_from_similarity(self, vectors=False, normalize=True, absolute_weights=False):
        """ the idea is to find the vocab(s) in the vocab-to-vocab similarity matrix (VV^T) that most
        closely matches the weights to each embedding dim ((VV^T)^TV=(VV^T)V=V(V^TV)), the idea being that if
        two words have close similarity to all words then they must be the same (if it walks like a duck...) """

        old_vocab_embeddings = self.vocab_embeddings
        if normalize:
            norms = np.linalg.norm(self._vocab_embeddings, axis=1, keepdims=True)
            self._vocab_embeddings = self._vocab_embeddings / norms
        try:
            if absolute_weights:
                PC_forward_emb = self.vocab_embeddings.T.dot(np.abs(self.vocab_embeddings))
            else:
                PC_forward_emb, _ = self.auto_name(vectors=True)  # = V^TV ~ vocab x vocab
            PC_similarity_to_words = self.vocab_embeddings.dot(PC_forward_emb)  # = VV^TV ~ vocab x emb
        finally:
            self._vocab_embeddings = old_vocab_embeddings
        vocab_indices = np.argmax(PC_similarity_to_words, axis=0)
        similarity = np.max(PC_similarity_to_words, axis=0)
        if vectors:
            names = self.vocab_embeddings[vocab_indices]
        else:
            names = self.dataset.vocab[vocab_indices]
        return names, similarity

    def neg_name(self):
        names_idx = np.argmin(self._vocab_embeddings, axis=0)
        names = self._vocab_embeddings[names_idx]
        return self.soft_inverse(names)

    def auto_name(self, vectors=False, forward=True, min_abstractness=None):
        """
        auto names the embeddings, if vectors is true it returns embedding vectors rather than words
        :return: names, cosine_similarity
        """
        if forward:
            names = network_helpers.auto_name(self.vocab_embeddings, W=self._embedding_weights[0],
                                              b=self._embedding_weights[1])
            names = names/np.linalg.norm(names, axis=1, keepdims=True)
        else:
            names_idx = np.argmax(self._vocab_embeddings, axis=0)
            names = self._vocab_embeddings[names_idx]

        if not vectors:
            names, similarity = self.soft_inverse(names, min_abstractness=min_abstractness)
        else:
            similarity = np.ones_like(names)

        return names, similarity

    def get_abstractness_scores(self, words, n_sample=1000):
        """ approximates concreteness of the words based on similarity distribution skew """
        vocab_embeddings = self.vocab_embeddings
        sample_idx = np.random.choice(vocab_embeddings.shape[0], n_sample)
        vocab_embeddings = vocab_embeddings[sample_idx]

        if issubclass(type(words[0]), str):
            word_emb = self.embed(words)
        else:
            word_emb = np.array(words)

        # consine similarity (since they are already normalized)
        similarity_matrix = vocab_embeddings.dot(word_emb.T)
        return scipy.stats.skew(similarity_matrix, axis=0)


# TODO: test me
class SS_Embedder(Embedder):
    """ SS = statisically significant, only uses vocab embeddings components
        that are deemed so (for predicting itself, kind of a weird concept) """

    @property
    def _vocab_embeddings(self):
        return self.__vocab_embeddings

    @_vocab_embeddings.setter
    def _vocab_embeddings(self, vocab_embeddings):
        """ the idea here is to only use statistically significant portions of vocab embeddings,
         it is hard to justify only using a subset under certain conditions """

        mask = self.F_test_contributions(vocab_embeddings.T)
        # F test is the same thing that happens in R for linear models

        self.__vocab_embeddings = (vocab_embeddings.T * mask).T

    def __init__(self, model_path=None, dataset: embedding_datasets.VocabEmbeddingDataset = None):
        super().__init__(model_path, dataset)
        self.__vocab_embeddings = None

    # based off of: https://stackoverflow.com/questions/21494141/how-do-i-do-a-f-test-in-python
    # and: http://www.real-statistics.com/multiple-regression/testing-significance-extra-variables-regression-model/
    # we are using first principles to understand F-test
    @classmethod
    def F_test_contributions(cls, weights, alpha=0.05, max_checked=1000):
        """returns mask of statistically significant weights"""

        embeddings = weights

        # eye = np.eye(embeddings.shape[0])
        mask = []

        for i, vec in enumerate(weights):
            contrib_rank_idx = np.argsort(np.abs(vec))[::-1]
            mask_i = np.array([False] * len(vec))

            SSE_reduced = np.sum((0 - embeddings[i]) ** 2)
            # prediction with no variables is always 0

            for idx in contrib_rank_idx[:max_checked]:
                mask_i[idx] = True
                vec_reduced = vec * mask_i
                preds = vec_reduced  # .dot(eye) == vec_reduced
                # approximation of embeddings (preds in lm terminology)

                SSE = np.sum((preds - embeddings[i]) ** 2)
                MSE = SSE / len(embeddings[i])

                F_stat = (SSE_reduced - SSE) / MSE  # looks like relative error? (except why is SSE not MSE??)
                df1 = df2 = len(embeddings[i]) - 1  # df's should be number of variables in each pred/loss vec (??)
                p_val = 1 - stats.f.cdf(F_stat, df1, df2)

                signif_contribution = p_val < alpha
                if not signif_contribution:  # if test fails
                    mask_i[idx] = False
                    break
                SSE_reduced = SSE  # this is reduced form of next model
            mask.append(mask_i)
        return np.stack(mask)
