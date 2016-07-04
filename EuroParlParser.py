"""
Author: Uriel Mandujano
Parser for eurparl data available at
    http://www.statmt.org/europarl/
For use with the Moses decoder

NOTE:
Moses decoder only compatible with:
    - all words must be lowercased
    - 1 sentence per line, no empty lines
    - sentences no longer than 100 words
"""
import os
import subprocess

import nltk.data

class EuroParlParser(object):
    def __init__(self, lang1_dir, lang2_dir):
        self.lang1_dir = lang1_dir
        self.lang2_dir = lang2_dir
        self._lang_files_exist()
        self._load_langs()

    def __str__(self):
        return "{}\n{}".format(self.lang1_dir, self.lang2_dir)

    def _lang_files_exist(self):
        assert os.path.exists(self.lang1_dir), \
            "{} file not found".format(self.lang1_dir)
        assert os.path.exists(self.lang2_dir), \
            "{} file not found".format(self.lang2_dir)

    def _load_langs(self):
        """
        Loads language data from files into class matrices with a
        correspondence between indices in each matrix. Lowercases words here
        """
        self.lang1, self.lang2 = [], []
        file1 = open(self.lang1_dir)
        file2 = open(self.lang2_dir)
        for line1, line2 in zip(file1, file2):
            line1 = self._strip_nonascii(line1.strip().lower())
            line2 = self._strip_nonascii(line2.strip().lower())
            self.lang1.append(line1), self.lang2.append(line2)

        assert len(self.lang1) == len(self.lang2), "Unequal language sizes"
        assert len(self.lang1) and len(self.lang2), "Got language of size 0"

    def _strip_nonascii(self, b):
        """
        Code to remove non-ascii characters from textfiles.
        Taken from jedwards on StackOverflow.
        """
        return b.decode('ascii', errors='ignore')

    def split_sentences(self):
        """
        Splits the language into sentences
        """
        #TODO
        #NOTE Tokenize before this to improve sentence splitting
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        lang1 = ''.join(self.lang1)
        lang2 = ''.join(self.lang2)
        print '\n-----\n'.join(tokenizer.tokenize(lang1))
        #print '\n-----\n'.join(tokenizer.tokenize(lang2))

    def clean_corpus(self):
        """
        Drop lines that are empty, too short, or too long.
        """
        min_line_len = 0
        max_line_len = 100
        pop_indices = []
        for i, (l1, l2) in enumerate(zip(self.lang1, self.lang2)):
            l1, l2 = l1.split(), l2.split()
            if l1 == l2 == []:
                pop_indices.append(i)
            elif len(l1) <= min_line_len or len(l2) <= min_line_len:
                pop_indices.append(i)
            elif len(l1) > max_line_len or len(l2) > max_line_len:
                pop_indices.append(i)

        for i in pop_indices[::-1]:
            self.lang1.pop(i), self.lang2.pop(i)

    def create_vocab(self):
        """
        Creates a dictionary containing the vocabulary of each language.
        Keys are words, values are counts
        """
        self.vocab_lang1, self.vocab_lang2 = dict(), dict()
        self._create_lang_vocab(self.vocab_lang1, self.lang1)
        self._create_lang_vocab(self.vocab_lang2, self.lang2)

    def _create_lang_vocab(self, vocab_lang, lang):
        """
        Populates language dictionary and each word entry's counts. Takes
        as parameter a dictionary{key:word; value:count} and lang matrix
        """
        for sentence in lang:
            sentence = sentence.split()
            for word in sentence:
                vocab_lang[word] = 1 + vocab_lang.get(word, 0)

    def vocab_to_list(self):
        """
        Returns a tuple with two entries: a list of the vocabulary in
        language 1 and another list of the vocabulary in language 2
        """
        return sorted(self.vocab_lang1.keys()), \
               sorted(self.vocab_lang2.keys())

def main():
    lang1 = '/Users/urielmandujano/data/europarl/europarl-v7.es-en.en'
    lang2 = '/Users/urielmandujano/data/europarl/europarl-v7.es-en.es'
    euro_parser = EuroParlParser(lang1, lang2)
    euro_parser.split_sentences()
    #euro_parser.clean_corpus()
    #euro_parser.create_vocab()
    #print euro_parser.vocab_to_list()[0]

if __name__ == '__main__':
    main()
