import numpy as np
from network_helpers import np_cos_similarity, np_diag_dot
import re


class VocabEmbeddingDataset:
    """ stores vocab to encode/decode words & one-hot-matrices/DTMs """

    def __init__(self, ordered_vocab=None, gensim_model=None):
        if gensim_model:
            # confirmed that index is index in one-hot-vectors!
            # https://stackoverflow.com/questions/63549977
            ordered_vocab = sorted(gensim_model.vocab.items(), key=lambda x: x[1].index)
            ordered_vocab = [vocab[0] for vocab in ordered_vocab]
        self._vocab = np.array(ordered_vocab)
        # assert (np.all(self.decode(self.encoded_vocab) == self.soft_decode(self.encoded_vocab)))

    @property
    def vocab(self):
        return self._vocab

    @property
    def one_hot_indices(self):  # dict of words to one-hot indices
        return {self.vocab[i]: i for i in range(len(self.vocab))}

    @property
    def encoded_vocab(self):  # returns an eye matrix
        return self.encode(self.vocab)

    # verified to work
    def encode(self, corpus):
        """ encodes corpus as n_corpus x n_vocab one_hot matrix """
        indices = self.one_hot_indices
        one_hot_mat = np.zeros([len(corpus), len(self.vocab)])  # one-hot encoding of corpus

        for i, word in enumerate(corpus):
            one_hot_mat[i, indices[word]] = 1
        return one_hot_mat

    def dual_name(self, row, target_mat):
        highest_sim = 0
        highest_pair = None

        for i in range(target_mat.shape[0]):
            for j in range(i, target_mat.shape[0]):
                mid_point = target_mat[i] + target_mat[j] / 2
                sim = np_cos_similarity(row, mid_point)
                if sim > highest_sim:
                    highest_sim = sim
                    highest_pair = (i, j)

        return highest_pair, highest_sim

    # TODO: combine soft decode with regular decode!
    def soft_decode(self, soft_dtm, _target_mat=None, acceptable_name=lambda x: True):
        """
        soft decodes based on distance metric
        :param soft_dtm: (dtm) to match against target_mat
        :param _target_mat: mat where each row corresponds to the target (exact)
         row value for a particular vocab word
        :param acceptable_name: lambda checking if a given name
         is acceptable (e.g. lambda x: len(x)>2)
        :param concreteness_scores: concreteness of each
        :return: words, cosine_similarities
        """

        if _target_mat is None:
            _target_mat = np.eye(len(self.vocab))

        assert (soft_dtm.shape[1] == _target_mat.shape[1])

        words = []
        similarity = []
        for i, row in enumerate(soft_dtm):
            sim = np_cos_similarity(_target_mat.T, row.T)

            # NOTE: distance is already a col vector
            ids = reversed(np.argsort(sim.flatten()))
            id = 0
            for i in ids:  # search for closest acceptable name and use that
                if acceptable_name(self.vocab[id]):
                    id = i
                    break

            words.append(self.vocab[id])
            similarity.append(sim[id])
        return words, similarity

    # verified to work
    def decode(self, one_hot_mat):
        corpus = []
        for i, row in enumerate(one_hot_mat):
            # returns a tuple of an array for some reason...
            ids = np.nonzero(row)[0]

            words = tuple([self.vocab[id] for id in ids])
            corpus.append(words if (len(words) > 1) else words[0])
        return corpus


class CorpusEmbeddingDataset(VocabEmbeddingDataset):
    """encodes/decodes words to dtm & back (i.e. one-hot matrix) & builds training data for embedding network"""

    def __init__(self, text_file=None, n_bytes=None, deletion_pattern=r"[#*&(:;’`\"\)0-9]"):
        with open(text_file) as f:
            corpus = f.read(n_bytes)
        corpus = re.sub(deletion_pattern, '', corpus.lower()).split()  # remove punctuation & split on whitespace

        super().__init__(np.array(list(set(corpus))))  # list of vocab (unique)
        self._word_one_hot = self.encode(corpus)  # just used to recompute skip gram
        self._gram_one_hot = None

    # verified to work
    def skip_gram_encoding(self, n, skip=1):
        """ sets the skip gram encoding settings """
        kernel_width = skip * (n - 1) + n  # n-1 is spaces in-between for skip, n is obv
        n_samples = self._word_one_hot.shape[0] - kernel_width  # stride is 1, so n_samples are the extra space
        new_mat = np.zeros([n_samples, self._word_one_hot.shape[1]])  # skip_gram encoding mat
        for i in range(n_samples):
            sample = self._word_one_hot[i:i + kernel_width:(skip + 1), :]  # the gram
            new_mat[i, :] = np.sum(sample, axis=0)  # make small DTM row (i.e. so doc=gram)
        self._gram_one_hot = new_mat
        assert(np.all(np.sum(self._gram_one_hot, axis=1) == n))

    def get_training_data(self, one_to_one=True):
        """splits DTM (of grams) into input and output data for training (guess context based on single word)"""
        one_hot = self._gram_one_hot
        input_data = np.zeros_like(one_hot)
        output_data = np.zeros_like(one_hot)
        for i, row in enumerate(one_hot):
            row = np.copy(row)

            # returns a tuple of an array for some reason...
            # (this makes row have just the one word in the gram)
            in_word_id = np.random.choice(np.nonzero(row)[0])
            row[in_word_id] = row[in_word_id] - 1  # don't let this word be reused
            input_data[i, in_word_id] = 1

            if one_to_one:
                out_word_id = np.random.choice(np.nonzero(row)[0])
                output_data[i, out_word_id] = 1
            else:
                output_data[i, :] = row

        print(self.decode(input_data[:10, :]))
        print(self.decode(output_data[:10, :]))
        return input_data, output_data

    def get_cooccurrence_matrix(self):
        grams = self._gram_one_hot

        cooccurrence_matrix = grams.T.dot(grams)

        # assert we have some non-zero non-diagonal elements as well...
        mask = np.logical_not(np.eye(len(self.vocab)))
        assert(len(np.nonzero(cooccurrence_matrix * mask)[0]) > 0)

        return cooccurrence_matrix

    def decode(self, one_hot_mat=None):
        if one_hot_mat is None:
            one_hot_mat = self._gram_one_hot
        return super().decode(one_hot_mat)
