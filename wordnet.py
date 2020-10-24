from nltk.corpus import wordnet as wn
from pprint import pprint
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
        print(
            "\nname: {} \ndef.: {} \ne.g.: {} \nlemmas: {}".format(
                syn.name(), syn.definition(), syn.examples(), syn.lemma_names()
            )
        )


def part_a(word):
    print('Number of synsets of "{}" are {}'.format(word, len(get_synsets(word))))
    print('Number of nouns on "{}" are {}'.format(word, len(get_synsets_noun(word))))
    print('"{}"\'s synsets description'.format(synset_description(get_synsets(word))))


"""
        PART B; WordNet taxonomic relations
"""

# 12 Hypernym chain
def filter_index(synset_list, index_list):
    return [syn for syn in synset_list for index in index_list if index in syn.name()]


def get_hypernyms_tree(synset_list):
    for syn in synset_list:
        print("\nHypernym chain for {} ({})".format(syn, syn.lemma_names()))
        hyp = lambda s: s.hypernyms()
        pprint(syn.tree(hyp))


def get_taxonomic_distance(synset_1, synset_2):
    print()
    print(
        "path_similarity between {} and {} is: {}".format(
            synset_1, synset_2, synset_1.path_similarity(synset_2)
        )
    )
    print(
        "lch_similarity between {} and {} is: {}".format(
            synset_1, synset_2, synset_1.lch_similarity(synset_2)
        )
    )
    print(
        "wup_similarity between {} and {} is: {}".format(
            synset_1, synset_2, synset_1.wup_similarity(synset_2)
        )
    )


def part_b(word):
    print("\nThis synsets were filtered to match the figure 19.5 of Jurafsky")
    synset_list = filter_index(synset_list=get_synsets(word), index_list=[".03", ".07"])
    get_hypernyms_tree(synset_list=synset_list)
    get_taxonomic_distance(synset_1=synset_list[0],synset_2=synset_list[1])


if __name__ == "__main__":
    # part_a(word="study")
    part_b(word="bass")


# TODO: part_b() print .lemma_names() not synset
