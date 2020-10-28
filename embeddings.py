import nltk
import numpy as np

#nltk.download('word2vec_sample')
path_to_word2vec_sample = nltk.data.find('models/word2vec_sample/pruned.word2vec.txt')

words,vectors, word2vec=[],[],{} #to collect all words and all vectors from the data sample.

with open(path_to_word2vec_sample) as file:
    for i, line in enumerate(file):
        if i == 0:
            continue
        line = line.split()
        word = line[0]
        vector = line[1:]
        vector = [float(s) for s in vector]

        words.append(word)
        vectors.append(vector)
        word2vec[word] = vector

print(words[:10])
print(len(words), len(vectors))

vectors = np.array(vectors)
print(vectors.shape)

print(word2vec["water"])

def norm(vector):
    return np.sqrt(sum([a*a for a in vector]))

print(word2vec["water"])

def dot(vector1,vector2):
    return sum([i * j for i,j in zip(vector1,vector2)])

def cosine(vector1,vector2):
    return dot(vector1,vector2) / (norm(vector1)*norm(vector2))

print(cosine(word2vec["dog"],word2vec["cat"]))