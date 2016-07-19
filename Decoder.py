"""
Author: Uriel Mandujano
Decoder for use with Moses and the Eurparl data. Performs statistical
MT between two pairs of languages and by pivoting.
"""
from EuroParlParser import EuroParlParser



class Decoder(object):
    def __init__(self, portion, lang1_dir, lang2_1_dir, lang2_2_dir ,lang3_dir):
        parser1 = EuroParlParser(lang1_dir, lang2_1_dir)
        lang1, lang2_1 = parser1.get_random_subset_corpus(portion)
        parser2 = EuroParlParser(lang2_2_dir, lang3_dir)
        lang2_2, lang3 = parser2.get_random_subset_corpus(portion)

def main():
    lang1 = '/Users/urielmandujano/data/europarl/europarl-v7.es-en.es'
    lang2_1 = '/Users/urielmandujano/data/europarl/europarl-v7.es-en.en'
    lang2_2 = '/Users/urielmandujano/data/europarl/europarl-v7.fr-en.en'
    lang3 = '/Users/urielmandujano/data/europarl/europarl-v7.fr-en.fr'
    decoder = Decoder(.1, lang1, lang2_1, lang2_2, lang3)

if __name__ == "__main__":
    main()
