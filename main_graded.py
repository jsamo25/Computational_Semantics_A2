import pandas as pd
import numpy as np
import nltk
import random
from nltk.corpus import wordnet as wn

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
#print(data.describe())

def get_pos_tag(string):
    #https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    tokens = nltk.word_tokenize(string)
    return nltk.pos_tag(tokens)[0][1]

data["target_word_pos"] = data["target_word"].apply(get_pos_tag)

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

#print(compare(data["baseline_random"],data["synset"]))
#print(compare(data["baseline_first"],data["synset"]))

#print(data[["target_word","synset","baseline_random","baseline_first","synset_is_correct"]][:5])
