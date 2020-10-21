from nltk.corpus import wordnet as wn

"""
        PART A; WordNet synsets and lemmas
"""
#3. how many wordnet synset does the word "study" has? how many are nouns?
print("All synsets: ", wn.synsets("study"))
print("the number of Nouns is:",len(wn.synsets("study",pos=wn.NOUN)))

#4 for every noun synset of study print the synset's name, definition and examples.
for syns in wn.synsets("study"):
    print("Name: {}, Definition: {}, Examples: {}".format(syns.name(), syns.definition(), syns.examples()))

#5 print all words the synset contains, i.e. the lemmas
for syns in wn.synsets("study"):
    print(syns.lemmas())