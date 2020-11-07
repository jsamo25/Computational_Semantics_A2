import nltk
import random
import numpy as np
import pandas as pd

from nltk.wsd import lesk
from itertools import chain
from nltk.corpus import wordnet as wn
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from pdb import set_trace

"""**********************************************************
        PART G: Word sense disambiguation: exploration
**********************************************************"""

data = pd.read_csv("semcor.csv")#[:100]
pd.set_option("display.max_columns", 20)

# 2 SemCor dataset statistics
def get_full_sentence(target_word, context_before, context_after):
    return str(context_before) + " " + str(target_word) + " " + str(context_after)

def get_pos_tag(string):
    tokens = nltk.word_tokenize(string)
    return nltk.pos_tag(tokens)[0][1]

data["full_sentence"] = data[["target_word", "context_before", "context_after"]].apply(lambda x: get_full_sentence(*x), axis=1)
data["len_sentence"] = data["full_sentence"].apply(lambda x: len(x))
data["target_word_pos"] = data["target_word"].apply(get_pos_tag)
# print(data["full_sentence"])
# print(data["target_word_pos"])
# print(data.describe())

# 3 baselines: random baseline and a baseline that always assigns the most common (1st) synset on yhe list.
def baseline_get_random_synset(word):
    synsets = wn.synsets(word)
    return random.choice(synsets).name()

def baseline_get_first_synset(word):
    synsets = wn.synsets(word)[::-1]
    return synsets.pop().name()

def baseline_get_last_synset(word):
    synsets = wn.synsets(word)
    return synsets.pop().name()

data["baseline_random"] = data["target_word"].apply(baseline_get_random_synset)
data["baseline_first"] = data["target_word"].apply(baseline_get_first_synset)
data["baseline_last"] = data["target_word"].apply(baseline_get_last_synset)

def compare(target, prediction):
    # used to compare with the baseline prediction, regardless the True/False column.
    correct = target == prediction
    return correct.mean()

print("Baseline random:", compare(target = data["synset"],prediction = data["baseline_random"]))
print("Baseline first:",compare(target = data["synset"],prediction = data["baseline_first"]))
print("Baseline last:",compare(target = data["synset"],prediction = data["baseline_last"]))

set_trace()
# print(data[["target_word","synset","baseline_random","baseline_first","synset_is_correct"]][:5])
# true_predicted = data.groupby(["target_word", "synset","baseline_random","baseline_first"])["synset_is_correct"].count()
# print(true_predicted)

"""**********************************************************
        PART H: WordNet and word embeddings
**********************************************************"""

try:
    path_to_word2vec_sample = nltk.data.find(
        "models/word2vec_sample/pruned.word2vec.txt"
    )
except:
    print("path not found, downloading from nltk...")
    nltk.download("word2vec_sample")
    path_to_word2vec_sample = nltk.data.find(
        "models/word2vec_sample/pruned.word2vec.txt"
    )

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

def get_word2vec(word):
    try:
        return word2vec[str(word)]
    except:
        return word2vec["bias"]

def cosine(vector1, vector2):
    return np.dot(vector1, vector2)/(np.linalg.norm(vector1) * np.linalg.norm(vector2))
# print(cosine(word2vec["love"],word2vec["hate"]))

def get_synsets(word):
    return wn.synsets(word)
# print(get_synsets("apples"))

def get_synonyms(word):
    synsets = get_synsets(word)
    return list(set(chain.from_iterable([syn.lemma_names() for syn in synsets])))
#print(get_synonyms("apples"))

def get_hypernyms(word):
    synsets = get_synsets(word)
    hypernyms = [syn.hypernyms() for syn in synsets]
    return list(
        set(
            chain.from_iterable(
                [hyp.lemma_names() for hyp in chain.from_iterable(hypernyms)]
            )
        )
    )
# print(get_hypernyms("apples"))

def get_hyponyms(word):
    synsets = get_synsets(word)
    hyponyms = [syn.hyponyms() for syn in synsets]
    return list(
        set(
            chain.from_iterable(
                [hyp.lemma_names() for hyp in chain.from_iterable(hyponyms)]
            )
        )
    )
# print(get_hyponyms("dog"))

def synonym_similarity(word):
    synonyms = get_synonyms(word)
    syn_similarity = [cosine(get_word2vec(syn), get_word2vec(word)) for syn in synonyms]
    return np.average(syn_similarity)

def hypernym_similarity(word):
    hypernyms = get_hypernyms(word)
    hyp_similarity = [cosine(get_word2vec(hyp), get_word2vec(word)) for hyp in hypernyms]
    return np.average(hyp_similarity)

def hyponym_similarity(word):
    hyponyms = get_hyponyms(word)
    hyp_similarity = [cosine(get_word2vec(hyp), get_word2vec(word)) for hyp in hyponyms]
    return np.average(hyp_similarity)

data["synonym_similarity"] = data["target_word"].apply(synonym_similarity)
data["hypernym_similarity"] = data["target_word"].apply(hypernym_similarity)
data["hyponym_similarity"] = data["target_word"].apply(hyponym_similarity)

# print("synonym_similarity:",data["synonym_similarity"].mean())
# print("hypernym_similarity:",data["hypernym_similarity"].mean())
# print("hyponym_similarity:",data["hyponym_similarity"].mean())

def sentence2vec(sentence):
    sentence = str(sentence).lower().split()
    sentence_embeddings = np.array([get_word2vec(word) for word in sentence])
    return np.mean(sentence_embeddings, axis=0)
# print(sentence2vec("i like rain"))

def sentence_cosine_similarity(sentence1, sentence2):
    return cosine(sentence2vec(sentence1), sentence2vec(sentence2))
# print(sentence_cosine_similarity(sentence1="all dogs go to heaven",sentence2="all cats go to hell")

def compare_most_common_definition(word):
    synsets = get_synsets(word)
    most_common_def = synsets[0].definition()
    return sentence_cosine_similarity(word, most_common_def)

def compare_less_common_definition(word):
    synsets = get_synsets(word)
    less_common_def = synsets[-1].definition()
    return sentence_cosine_similarity(word, less_common_def)

data["most_common_def_similarity"] = data["target_word"].apply(compare_most_common_definition)
data["less_common_def_similarity"] = data["target_word"].apply(compare_less_common_definition)
# print("Word compared to it's most common def.",data["most_common_def_similarity"].mean())
# print("Word compared to it's less common def.",data["less_common_def_similarity"].mean())

"""**********************************************************
             PART I: Word sense disambiguation
**********************************************************"""

def count_word_overlap(pred_synset,sentence):
    synset = wn.synset(pred_synset)
    return len(set(sentence.split()) & set(synset.definition().split()))

def synset_name_and_sentence_similarity(pred_synset, sentence):
    synset = wn.synset(pred_synset)
    return sentence_cosine_similarity(synset.lemma_names()[0], sentence)

def synset_and_target_lemmas(pred_synset, word):
    synset = wn.synset(pred_synset)
    return cosine(get_word2vec(synset.lemma_names()[0]), get_word2vec(word))

def target_word_and_synset_definition(word, pred_synset):
    synset = wn.synset(pred_synset)
    return sentence_cosine_similarity(synset.definition(), word)

def synset_definition_and_sentence(pred_synset, sentence):
    synset = wn.synset(pred_synset)
    return sentence_cosine_similarity(synset.definition(), sentence)

def synset_example_and_sentence(pred_synset, sentence):
    synset = wn.synset(pred_synset)
    try:
        return sentence_cosine_similarity(synset.examples()[0], sentence)
    except:
        return sentence_cosine_similarity("bias", sentence)

def lesk_algorithm(word, context):
    return lesk(context.split(), word)  # .name()

def target_word_and_lesk_similarity(word, context):
    lesk = lesk_algorithm(word, context).lemma_names()[0]
    return sentence_cosine_similarity(lesk, word)

#TODO: Synset frequency: extract synset index from list and use it as feature

data["f0_overlapping_words"] = data[["synset","full_sentence"]].apply(lambda x: count_word_overlap(*x), axis=1)
data["f1_synset_and_context_after"] = data[["synset", "context_after"]].apply(lambda x: synset_name_and_sentence_similarity(*x), axis=1)
data["f2_synset_and_context_before"] = data[["synset", "context_before"]].apply(lambda x: synset_name_and_sentence_similarity(*x), axis=1)
data["f3_synset_and_target_similarity"] = data[["synset", "target_word"]].apply(lambda x: synset_and_target_lemmas(*x), axis=1)
data["f4_target_and_synset_definition"] = data[["target_word", "synset"]].apply(lambda x: target_word_and_synset_definition(*x), axis=1)
data["f5_synset_and_sentence_similarity"] = data[["synset", "full_sentence"]].apply(lambda x: synset_name_and_sentence_similarity(*x), axis=1)
data["f6_target_word_and_sentence_similarity"] = data[["target_word", "full_sentence"]].apply(lambda x: sentence_cosine_similarity(*x), axis=1)
data["f7_synset_example_and_sentence_similarity"] = data[["synset", "full_sentence"]].apply(lambda x: synset_example_and_sentence(*x), axis=1)
data["f8_synset_definition_and_sentence_similarity"] = data[["synset", "full_sentence"]].apply(lambda x: synset_definition_and_sentence(*x), axis=1)
#data["f9_lesk_pred_and_target_similarity"] = data[["target_word", "full_sentence"]].apply(lambda x: target_word_and_lesk_similarity(*x), axis=1)

data_train, data_test = train_test_split(data, test_size=0.20, random_state=1)
feature_list= [
            "f0_overlapping_words",
            "f1_synset_and_context_after",
            "f2_synset_and_context_before",
            "f3_synset_and_target_similarity",
            "f4_target_and_synset_definition",
            "f5_synset_and_sentence_similarity",
            "f6_target_word_and_sentence_similarity",
            "f7_synset_example_and_sentence_similarity",
            "f8_synset_definition_and_sentence_similarity",
            #"f9_lesk_pred_and_target_similarity",  # not useful
        ]
y_train, y_test = data_train["synset_is_correct"], data_test["synset_is_correct"]
x_train, x_test = (data_train[feature_list],data_test[feature_list])

def accuracy(model, x_train, y_train, x_test, y_test):
    print("training set:", model.score(x_train, y_train))
    print("testing set:", model.score(x_test, y_test))

def compare(target, prediction):
    correct = target == prediction
    return correct.mean()

model = LogisticRegression(C=10,class_weight='balanced').fit(x_train, y_train)
print("\nInitial model Coefficients\n", model.coef_.squeeze())
print("model accuracy:")
accuracy(model, x_train, y_train, x_test, y_test)

data["baseline_ran"] = np.random.choice([True, False], size=len(data), p=[0.5, 1 - 0.5])
data["baseline_pos"] = True
data["baseline_neg"] = False

print("\n Compare to baselines:")
print("accuracy of all-pos baseline",compare(data["synset_is_correct"], data["baseline_pos"]))
print("accuracy of all-neg baseline",compare(data["synset_is_correct"], data["baseline_neg"]))
print("accuracy of all-ran baseline",compare(data["synset_is_correct"], data["baseline_ran"]))

"""**********************************************************
                 PART J: Extra points
**********************************************************"""

#print(data.describe())


