"""
Author: Uriel Mandujano
Parser for eurparl data available at
    http://www.statmt.org/europarl/
For use with the Moses decoder
"""
import os
import subprocess

class EuroParlParser(object):
    def __init__(self, lang1, lang2):
        self.lang1 = lang1
        self.lang2 = lang2
        self.lang_files_exist()

    def __str__(self):
        return "{}\n{}".format(self.lang1, self.lang2)

    def lang_files_exist(self):
        assert os.path.exists(self.lang1), \
            "{} file not found".format(self.lang1)
        assert os.path.exists(self.lang2), \
            "{} file not found".format(self.lang2)

    def clean_corpus(self):
        """
        Remove empty lines, remove redundant space characters, drop lines
        that are too empty, short, or long
        """
        command = 'clean-corpus-n.perl CORPUS L1 L2 OUT MIN MAX'
        #TODO

    def sentence_align(self):
        """
        Align sentences between the two corpuses using the Gale-Church
        algorithm. Recommended to improve translation quality as sentence-to-
        sentence translations are more reliable than paragraph to paragraph
        """
        #TODO

def main():
    lang1 = '/Users/urielmandujano/data/europarl/europarl-v7.es-en.en'
    lang2 = '/Users/urielmandujano/data/europarl/europarl-v7.es-en.es'
    euro_parser = EuroParlParser(lang1, lang2)
    print "Pass"

if __name__ == '__main__':
    main()
