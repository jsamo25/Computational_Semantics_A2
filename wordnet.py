import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from pdb import set_trace

"""
        PART A; WordNet synsets and lemmas
"""

def get_synsets(word):
    return wn.synsets(word)

def get_synsets_noun(word):
    return wn.synsets(word, pos=wn.NOUN)

def synset_description(synset_list):
    for syn in synset_list:
        print("\nname: {} \ndef.: {} \ne.g.: {} \nlemmas: {}".format(
                syn.name(), syn.definition(), syn.examples(), syn.lemma_names()))

def part_a(word):
    print('Number of synsets of "{}" are {}'.format(word, len(get_synsets(word))))
    print('Number of nouns on "{}" are {}'.format(word, len(get_synsets_noun(word))))
    print('"{}"\'s synsets description'.format(synset_description(get_synsets(word))))

"""
        PART B; WordNet taxonomic relations
"""


def filter_index(synset_list, index_list):
    return [syn for syn in synset_list for index in index_list if index in syn.name()]


def get_hypernyms_tree(synset_list):
    for syn in synset_list:
        print("\nHypernym chain for {} ({})".format(syn, syn.lemma_names()))
        hyp = lambda s: s.hypernyms()
        pprint(syn.tree(hyp))

def get_hypernyms_path(synset_list):
    for syn in synset_list:
        print("\nHypernym path for {} ({})".format(syn, syn.hypernym_paths()))

def get_taxonomic_distance(synset_1, synset_2):
    print()
    print(
        "path_similarity between {} and {} is: {}".format(
            synset_1, synset_2, synset_1.path_similarity(synset_2)))
    print(
        "lch_similarity between {} and {} is: {}".format(
            synset_1, synset_2, synset_1.lch_similarity(synset_2)))
    print(
        "wup_similarity between {} and {} is: {}".format(
            synset_1, synset_2, synset_1.wup_similarity(synset_2)))

def part_b(word):
    print("\nThis synsets were filtered to match the figure 19.5 of Jurafsky")
    synset_list = filter_index(synset_list=get_synsets(word), index_list=[".03", ".07"])
    get_hypernyms_tree(synset_list=synset_list)
    #get_hypernyms_path(synset_list=synset_list) #TODO: alternative method
    get_taxonomic_distance(synset_1=synset_list[0], synset_2=synset_list[1])


"""
        PART C; Word sense disambiguation
"""

pd.set_option("display.max_columns", 10)
from nltk.wsd import lesk

def load_data(csv_file="semcor30.csv"):
    data = pd.read_csv(csv_file)
    return data

def context_sentence(word, context_before, context_after):
    return context_before + " " + word + " " + context_after

def lesk_algorithm(context, word):
    return lesk(context.split(),word)#.name()

def data_new_features():
    data = load_data()
    data["full_sentence"] = data[["target_word", "context_before", "context_after"]].apply(lambda x: context_sentence(*x), axis=1)
    data["lesk_prediction"] = data[["full_sentence", "target_word"]].apply(lambda x: lesk_algorithm(*x), axis=1)
    return data

def word_text_disambiguation_leks(line):
    data = data_new_features()

    print("target word: ",data["target_word"][line])
    print("context: ",data["full_sentence"][line])
    print("\nlesk prediction: ",data["lesk_prediction"][line])
    print("definition: ",data["lesk_prediction"][line].definition())
    print("\ndata synset ref.: ",data["synset"][line])
    print("data synset is correct?", data["synset_is_correct"][line])
    print("\npossible synsets:")
    pprint(wn.synsets(data["target_word"][line]))

def part_c():
    word_text_disambiguation_leks(line=3)




if __name__ == "__main__":
    #part_a(word="study")
    #part_b(word="bass")
    part_c()
