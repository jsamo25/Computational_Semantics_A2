import nltk
import numpy as np

from collections import Counter
from pprint import pprint
from pdb import set_trace

"""
        PART D; Word vectors
"""

try:
    path_to_word2vec_sample = nltk.data.find('models/word2vec_sample/pruned.word2vec.txt')
except:
    print("path not found, downloading from nltk...")
    nltk.download('word2vec_sample')
    path_to_word2vec_sample = nltk.data.find('models/word2vec_sample/pruned.word2vec.txt')


words = []
vectors = []
word2vec = {} #to collect all words and all vectors from the data sample.

with open(path_to_word2vec_sample) as f:
    for i, line in enumerate(f):
        if i == 0:
            continue
        line = line.split()
        word = line[0]
        vector = [float(s) for s in line[1:]]

        words.append(word)
        vectors.append(vector)
        word2vec[word] = vector
#print("word2vec instantiated\n")
# print(words[:10])
# print(len(words), len(vectors))
# print(vectors.shape)
#print(word2vec["water"])

def norm(vector):
    return np.sqrt(sum([a*a for a in vector]))

def dot(vector1,vector2):
    return sum([i * j for i,j in zip(vector1,vector2)])

def cosine(vector1,vector2):
    return dot(vector1,vector2) / (norm(vector1)*norm(vector2))
#print(cosine(word2vec["cat"],word2vec["dog"]))

def get_cosine_similarity(word1,word2):
    print("The cosine similarity of {} and {} is:".format(word1,word2))
    return cosine(word2vec[word1],word2vec[word2])
#print(get_cosine_similarity("apple","banana"))
#vectors = np.array(vectors)

def get_cosine_pairs(target):
    cosine_pairs = {}
    for word in words:
        if word == target:
            continue #remove comparison with self
        cosine_pairs[target,word] = cosine(word2vec[target],word2vec[word])
    return cosine_pairs

def get_n_similar_words(target,n):
    k = Counter(get_cosine_pairs(target))
    most_common = k.most_common(n)
    print("{} most (cosine) similar pairs:".format(n))
    pprint(most_common)
#get_n_similar_words("dog", n = 10)