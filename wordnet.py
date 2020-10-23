from nltk.corpus import wordnet as wn
from itertools import chain
from pdb import set_trace

"""
        PART A; WordNet synsets and lemmas
"""
# 3 print number of synsets of a word, and number of NOUNS
word_synsets = wn.synsets("study")
word_synsets_noun = wn.synsets("study", pos=wn.NOUN)

print(
    'The Number of synsets for "study" is {} from which {} are nouns'.format(
        len(word_synsets), len(word_synsets_noun)
    )
)
# alt: print(sum(['.n.' in study_synset.name() for study_synset in study_synsets]))

# 4, #5 for every noun synset of study print the synset's name, definition, examles an lema names.
for syn in word_synsets_noun:
    print(
        "\nName: {} \nDef: {} \nExamples: {} \nLemmas: {}".format(
            syn.name(), syn.definition(), syn.examples(), syn.lemma_names()
        )
    )
# print([syn.lemma_names() for syn in study_synsets])

"""
        PART B; WordNet taxonomic relations
"""

# 12 Hyponymy chain
bass_syns = wn.synsets("bass")
bass_syns_book = [syn for syn in bass_syns if ".03" in syn.name() or ".07" in syn.name()]

for syn in bass_syns_book:
    print(
        "\nName: {} \nDef: {} \nExamples: {} \nLemmas: {}".format(
            syn.name(), syn.definition(), syn.examples(), syn.lemma_names()
        )
    )

print(bass_syns_book)
print(bass_syns_book[0].lemma_names())
print(bass_syns_book[1].lemma_names())

def extract_hypernyms (synset):
    return [word.lemma_names() for word in synset.hypernyms()][0]

def request_hypernyms (syn_list):
    for syn in syn_list:
        print(extract_hypernyms(syn))

request_hypernyms(bass_syns_book)
