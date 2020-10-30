import nltk

from gensim.similarities.index import  AnnoyIndexer
from gensim.models import Word2Vec


from pdb import set_trace

"""
        PART D; Word vectors (Gensim)
"""

def most_similar(sentence1,sentence2,target_word):
    sentences = [sentence1.split(),sentence2.split()]
    model = Word2Vec(sentences, min_count=1, seed=1)
    indexer = AnnoyIndexer(model,2)
    return model.wv.most_similar(target_word, topn=3, indexer=indexer)

print(
    most_similar(sentence1="flowers are red",
                 sentence2="violets are blue",
                 target_word="violets")
)
