import pandas as pd
import numpy as np
import nltk
import random

from nltk.corpus import wordnet as wn
from itertools import chain
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from pdb import set_trace

#nltk.download('averaged_perceptron_tagger')

"""
        PART G; Word sense disambiguation: exploration
"""

data = pd.read_csv("semcor30.csv")
pd.set_option("display.max_columns", 15)

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
    return np.average(syn_similarity)

def hypernym_similarity(word):
    hypernyms=get_hypernyms(word)
    hyp_similarity = [cosine(get_word2vec(hyp),get_word2vec(word)) for hyp in hypernyms]
    return np.average(hyp_similarity)

def hyponym_similarity(word):
    hyponyms=get_hyponyms(word)
    hyp_similarity = [cosine(get_word2vec(hyp),get_word2vec(word)) for hyp in hyponyms]
    return np.average(hyp_similarity)

data["synonym_similarity"] = data["target_word"].apply(synonym_similarity)
data["hypernym_similarity"] = data["target_word"].apply(hypernym_similarity)
data["hyponym_similarity"] = data["target_word"].apply(hyponym_similarity)

#(b) on average a word is more similar to it's hypernyms than to its hyponyms.

# print("synonym_similarity:",data["synonym_similarity"].mean())
# print("hypernym_similarity:",data["hypernym_similarity"].mean())
# print("hyponym_similarity:",data["hyponym_similarity"].mean())

#5 It is possible to use word embeddings for constructing a reasonable meaning
# representation also for a longer sequence of words, such as a sentence. Implement
# a function that does this, and use it to test the following hypothesis using
# word2vec embeddings: on average, a word is more similar to the definition of its
# most frequent sense (i.e., first synset), than to definitions of its less frequent
# senses.
def sentence2vec(sentence):
    sentence = sentence.lower().split()
    sentence_embeddings = np.array([get_word2vec(word) for word in sentence])
    return np.average(sentence_embeddings, axis=0)
#print(sentence2vec("i like rain"))

def sentence_cosine_similarity(sentence1,sentence2):
    return cosine(sentence2vec(sentence1),sentence2vec(sentence2))

def compare_most_common_definition(word):
    synsets=get_synsets(word)
    most_common_def = synsets[0].definition()
    return sentence_cosine_similarity(word,most_common_def)

def compare_less_common_definition(word):
    synsets =get_synsets(word)
    less_common_def = synsets[-1].definition()
    return sentence_cosine_similarity(word,less_common_def)

data["most_common_def_similarity"] = data["target_word"].apply(compare_most_common_definition)
data["less_common_def_similarity"] = data["target_word"].apply(compare_less_common_definition)

# print("Word compared to it's most common def.",data["most_common_def_similarity"].mean())
# print("Word compared to it's less common def.",data["less_common_def_similarity"].mean())

"""
        PART I; Word sense disambiguation
"""

def synset_and_target_lemmas(pred_synset, word):
    synset = wn.synset(pred_synset)
    return cosine(get_word2vec(synset.lemma_names()[0]),get_word2vec(word))

def sentence_and_target_definition(pred_synset,sentence):
    synset = wn.synset(pred_synset)
    return sentence_cosine_similarity(synset.lemma_names()[0],sentence)

def synset_definition_and_sentence(pred_synset, sentence):
    synset = wn.synset(pred_synset)
    return sentence_cosine_similarity(synset.definition(),sentence)

def synset_example_and_sentence(pred_synset,sentence):
    synset = wn.synset(pred_synset)
    try:
        return sentence_cosine_similarity(synset.examples()[0],sentence)
    except:
        return sentence_cosine_similarity("bias",sentence)


data["f1_synset_and_target_similarity"] = data[["synset","target_word"]].apply(lambda x: synset_and_target_lemmas(*x), axis=1)
data["f2_synset_and_sentence_similarity"] = data[["synset","full_sentence"]].apply(lambda x: sentence_and_target_definition(*x),axis=1)
data["f3_synset_definition_and_sentence_similarity"] = data[["synset","full_sentence"]].apply(lambda x: synset_definition_and_sentence(*x),axis=1)
data["f4_synset_example_and_sentence_similarity"] = data[["synset","full_sentence"]].apply(lambda x: synset_example_and_sentence(*x),axis=1)

data_train, data_test = train_test_split(data, test_size=0.3)
y_train, y_test = data_train["synset_is_correct"], data_test["synset_is_correct"]
x_train, x_test = (
    data_train[
        ["f1_synset_and_target_similarity",
         "f2_synset_and_sentence_similarity",
         "f3_synset_definition_and_sentence_similarity",
         "f4_synset_example_and_sentence_similarity"]
    ],
    data_test[
        ["f1_synset_and_target_similarity",
         "f2_synset_and_sentence_similarity",
         "f3_synset_definition_and_sentence_similarity",
         "f4_synset_example_and_sentence_similarity"]
    ],
)
def accuracy(model, x_train, y_train, x_test, y_test):
    print("training set:", model.score(x_train, y_train))
    print("testing set:", model.score(x_test, y_test))


# Logistic regression basic.
print("\n Initial model score")
model = LogisticRegression().fit(x_train, y_train)
print("\n Initial model Coefficients", model.coef_.squeeze())

print("model accuracy:")
accuracy(model, x_train, y_train, x_test, y_test)

