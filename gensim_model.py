import pandas as pd
import nltk
import re

from matplotlib import pyplot
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from pdb import set_trace

pd.set_option("display.max_columns", 10)
pd.set_option('display.width', 100)
pd.set_option("max_colwidth", 100)

data = pd.read_csv("data/semcor.csv")#[:1000]
STOP_WORDS = nltk.corpus.stopwords.words()

def get_full_sentence(target_word, context_before, context_after):
    return str(context_before) + " " + str(target_word) + " " + str(context_after)

def clean_sentence(context):
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub('', context).lower()
    sentence = sentence.split(" ")

    for word in list(sentence):
        if word in STOP_WORDS:
            sentence.remove(word)

    sentence = " ".join(sentence)
    return sentence

data["full_sentence"] = data[["target_word", "context_before", "context_after"]].apply(lambda x: get_full_sentence(*x), axis=1)
data["full_sentence"] = data["full_sentence"].apply(clean_sentence)

#tokenize and define corpus: full context from semcor.csv
corpus = [sentence.split() for sentence in data["full_sentence"]]

#train the model
model = Word2Vec(corpus, min_count=50)
print(model)
# print(list(model.wv.vocab))
# print(model["Friday"])

#reduce dimensionatlity to a 2d PCA model
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

#create the graph
pyplot.scatter(result[:,0],result[:,1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i,0], result[i,1]))

pyplot.show()

#print(model.wv.vocab)
#print(model.wv.most_similar("antenna"))
target_word="antenna"
print("most similar words to: {}".format([key[0] for key in model.wv.most_similar(str(target_word))]))
set_trace()