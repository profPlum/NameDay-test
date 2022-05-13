from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
import gensim.downloader as api
from embedders.word2vec_embedder import Word2VecEmbedder
from embedders import keras_embedder, embedding_datasets
from correlation_test import *
import pandas as pd
import numpy as np
import os
import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')

# fixes weird OMP mac bug
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

gensim_emb_name = "glove-wiki-gigaword-50" # 'glove-twitter-25' 
gensim_fn = './data/{}.vk'.format(gensim_emb_name)

# we only want to do this once
if os.path.exists(gensim_fn):
    gensim_model = KeyedVectors.load(gensim_fn)  # mmap makes this significantly faster (lazy loading)
else:
    gensim_model = api.load(gensim_emb_name)
    gensim_model.save(gensim_fn)

gensim_embedder = Word2VecEmbedder(model=gensim_model)

def write_lexicon(words, fn):
    defs = []
    for word in words:
        synsets = wn.synsets(word)
        if len(synsets) > 0:
            word = synsets[0]
            defs.append(word.definition())   
        else:
            defs.append(None)
    lexicon = pd.DataFrame({"Word": words, 'Definition': defs})
    lexicon.to_csv(fn, index=False)

def compare_soft_vs_true_names(embedder, true_PC_name_vecs=None):
    """
    Compares soft vs true names for auto names (although it should work 
    with any approximated embedding vectors)
    """
    import matplotlib.pyplot as plt

    # load PCA from cache otherwise compute it & save in cache
    if embedder in compare_soft_vs_true_names._pcas:
        pca = compare_soft_vs_true_names._pcas[embedder]
    else:
        pca = PCA(2)
        pca.fit(embedder.vocab_embeddings)
        compare_soft_vs_true_names._pcas[embedder] = pca
    
    if true_PC_name_vecs is None:
        true_PC_name_vecs, _ = embedder.auto_name(vectors=True)
    soft_PC_names, _ = embedder.soft_inverse(true_PC_name_vecs)
    soft_PC_name_vecs = embedder.embed(soft_PC_names)
    soft_plot_PCs = pca.transform(soft_PC_name_vecs)
    true_plot_PCs = pca.transform(true_PC_name_vecs)

    all_PCs = np.concatenate([soft_plot_PCs, true_plot_PCs])
    xy_min = np.min(all_PCs, axis=0)
    xy_max = np.max(all_PCs, axis=0)
    plt.axis([xy_min[0], xy_max[0], xy_min[1], xy_max[1]])
    
    for soft_point, true_point in zip(soft_plot_PCs, true_plot_PCs):
        dpoint =  soft_point - true_point  # difference from base (e.g. dx,dy)
        plt.arrow(true_point[0], true_point[1], dpoint[0], dpoint[1], length_includes_head=False,
          head_width=0.035, head_length=0.035)
    
    # we want to show the drift from name approximation
    plt.title('true --> soft names plot')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.savefig('logs/true --> soft names plot.png')

# initialize PCA cache
compare_soft_vs_true_names._pcas = {}

compare_soft_vs_true_names(gensim_embedder)

forward_PC_names, _ = gensim_embedder.auto_name(forward=True, vectors=True)
correlation_test(gensim_embedder, forward_PC_names)

# the vocab used for further experiments is restricted
common_vocab = pd.read_csv('./data/unigram_freq.csv')
whitelist_vocab = common_vocab['word'][:50000]
gensim_embedder.restrict_vocab(whitelist_vocab)

np.savetxt('./logs/vocab_emb.txt', gensim_embedder.vocab_embeddings)
write_lexicon(gensim_embedder.dataset.vocab, './logs/lexicon.csv')
write_lexicon(gensim_embedder.auto_name()[0], './logs/PC_names.csv')
