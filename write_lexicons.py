from PyDictionary import PyDictionary as pyDict
from gensim.models import KeyedVectors
import gensim.downloader as api
from embedders.word2vec_embedder import Word2VecEmbedder
import pandas as pd
import numpy as np
import os

gensim_emb_name = "glove-wiki-gigaword-50" # 'glove-twitter-25' 
gensim_fn = f'./data/{gensim_emb_name}.vk'

# we only want to do this once
if os.path.exists(gensim_fn):
    gensim_model = KeyedVectors.load(gensim_fn)  # mmap makes this significantly faster (lazy loading)
else:
    gensim_model = api.load(gensim_emb_name)
    gensim_model.save(gensim_fn)

gensim_embedder = Word2VecEmbedder(model=gensim_model)

def write_lexicon(words, fn):
    if type(words) is np.ndarray:
        words = words.tolist()
    
    dictionary = pyDict(words)
    form_names = ['Noun', 'Verb', 'Adjective', 'Adverb']
    lexicon = pd.DataFrame(columns=['Word'] + form_names)

    all_definitions = dictionary.getMeanings()
    
    # replace Nones with empty dicts to handle special cases properly
    for word in words:
        definitions = {form: None for form in form_names}
        new_defs = all_definitions[word]
        
        # always select the first definition to avoid complexity
        def simplify_defs(x):
            if type(x) is list:
                return x[0]
            else: return x
        
        if new_defs is None:
            new_defs = {} # we want a default of an empty dict not None for the sake of retaining entries for undefined words
        else: new_defs = {key: simplify_defs(val) for key, val in new_defs.items() if len(val)>0}
        
        definitions.update(new_defs)
        definitions['Word'] = word
        lexicon = lexicon.append(definitions, ignore_index=True)
    lexicon.to_csv(fn, index=False)

np.savetxt('./logs/vocab_emb.txt', gensim_embedder.vocab_embeddings)

write_lexicon(gensim_embedder.dataset.vocab, './logs/lexicon.csv')
write_lexicon(gensim_embedder.auto_name()[0], './logs/PC_names.csv')
