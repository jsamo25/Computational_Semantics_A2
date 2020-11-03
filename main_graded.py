import pandas as pd
import numpy as np
import nltk
import random

from nltk.corpus import wordnet as wn
from itertools import chain
from pdb import set_trace

#nltk.download('averaged_perceptron_tagger')

"""
        PART G; Word sense disambiguation: exploration
"""

data = pd.read_csv("semcor30.csv")
pd.set_option("display.max_columns", 10)

#2 SemCor dataset statistics
def get_full_sentence(target_word, context_before, context_after):
    return str(context_before) + " " + str(target_word) + " " + str(context_after)

data["full_sentence"] = data[["target_word", "context_before", "context_after"]].apply(lambda x: get_full_sentence(*x), axis=1)
data["len_sentence"]= data["full_sentence"].apply(lambda x: len(x))
#print(data["full_sentence"])
#print(data.describe())

def get_pos_tag(string):
    #https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    tokens = nltk.word_tokenize(string)
    return nltk.pos_tag(tokens)[0][1]

data["target_word_pos"] = data["target_word"].apply(get_pos_tag)
#print(data["target_word_pos"])

#3 baselines: random baseline and a baseline that always assigns the most common (1st) synset on yhe list.

def baseline_get_random_synset(word):
    synsets = wn.synsets(word)
    return random.choice(synsets).name()

def baseline_get_first_synset(word):
    synsets = wn.synsets(word)
    return synsets.pop().name()

data["baseline_random"] = data["target_word"].apply(baseline_get_random_synset)
data["baseline_first"] = data["target_word"].apply(baseline_get_first_synset)

def compare(target, prediction):
    #used to compare with the baseline prediction, regardless the True/False column.
    correct = target == prediction
    return correct.mean()

#TODO: define a better metric

# print(compare(target = data["synset"],prediction = data["baseline_random"]))
# print(compare(target = data["synset"],prediction = data["baseline_first"]))

# print(data[["target_word","synset","baseline_random","baseline_first","synset_is_correct"]][:5])
# true_predicted = data.groupby(["target_word", "synset","baseline_random","baseline_first"])["synset_is_correct"].count()
# print(true_predicted)

"""
        PART H; WordNet and word embeddings
"""
#define the Word2Vec Space
try:
    path_to_word2vec_sample = nltk.data.find('models/word2vec_sample/pruned.word2vec.txt')
except:
    print("path not found, downloading from nltk...")
    nltk.download('word2vec_sample')
    path_to_word2vec_sample = nltk.data.find('models/word2vec_sample/pruned.word2vec.txt')

words = vectors = []
word2vec = {}

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

#Cosine similarity functions
def cosine(vector1,vector2):
    return np.dot(vector1,vector2) / (np.linalg.norm(vector1)*np.linalg.norm(vector2))
#print(cosine(word2vec["love"],word2vec["hate"]))

def get_synsets(word):
    return wn.synsets(word)
#print(get_synsets("apples"))

def get_synonyms(word):
    synsets=get_synsets(word)
    return list(set(chain.from_iterable([syn.lemma_names() for syn in synsets])))
#print(get_synonyms("apples"))

def get_hypernyms(word):
    synsets=get_synsets(word)
    hypernyms=[syn.hypernyms() for syn in synsets]
    return list(set(chain.from_iterable([hyp.lemma_names() for hyp in chain.from_iterable(hypernyms)])))
#print(get_hypernyms("apples"))

def get_hyponyms(word):
    synsets=get_synsets(word)
    hyponyms=[syn.hyponyms() for syn in synsets]
    return list(set(chain.from_iterable([hyp.lemma_names() for hyp in chain.from_iterable(hyponyms)])))
#print(get_hyponyms("dog"))

def get_word2vec(word):
    try:
        return word2vec[str(word)]
    except:
        return word2vec["bias"]

def synonym_similarity(word):
    synonyms=get_synonyms(word)
    syn_similarity = [cosine(get_word2vec(syn),get_word2vec(word)) for syn in synonyms]
    return sum(syn_similarity)/len(syn_similarity)

def hypernym_similarity(word):
    hypernyms=get_hypernyms(word)
    hyp_similarity = [cosine(get_word2vec(hyp),get_word2vec(word)) for hyp in hypernyms]
    return sum(hyp_similarity)/len(hyp_similarity)

def hyponym_similarity(word):
    hyponyms=get_hyponyms(word)
    hyp_similarity = [cosine(get_word2vec(hyp),get_word2vec(word)) for hyp in hyponyms]
    return sum(hyp_similarity)/len(hyp_similarity)

data["synonym_similarity"] = data["target_word"].apply(synonym_similarity)
data["hypernym_similarity"] = data["target_word"].apply(hypernym_similarity)
data["hyponym_similarity"] = data["target_word"].apply(hyponym_similarity)

#(b) on average a word is more similar to it's hypernyms than to its hyponyms.

print("synonym_similarity:",data["synonym_similarity"].mean())
print("hypernym_similarity:",data["hypernym_similarity"].mean())
print("hyponym_similarity:",data["hyponym_similarity"].mean())

