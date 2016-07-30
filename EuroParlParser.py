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

1) Tokenize
2) Lowercase, remove > 100 word lines, 1 sentence per line, clear non-ascii

    ? add statistical methods
"""
import os
import subprocess

import random

import utils

NUM_CPUS = 4

class EuroParlParser(object):
    def __init__(self, lang1_dir, lang2_dir):
        self.lang1_dir = lang1_dir
        self.lang2_dir = lang2_dir
        self._check_lang_files_exist()

        if utils.data_exists('cleansed', self.lang1_dir, self.lang2_dir):
            pass
        elif utils.data_exists('tok', self.lang1_dir, self.lang2_dir):
            self._load_tokenized_data()
            self._clean_corpus()
        else:
            self._tokenize()
            self._load_tokenized_data()
            self._clean_corpus()

        self._load_cleansed_data()

    def __str__(self):
        return "{}\n{}".format(self.lang1_dir, self.lang2_dir)

    def _check_lang_files_exist(self):
        assert os.path.exists(self.lang1_dir), \
            "{} file not found".format(self.lang1_dir)
        assert os.path.exists(self.lang2_dir), \
            "{} file not found".format(self.lang2_dir)

    def _tokenize(self):
        """
        Given a list of sentences, we tokenize using the mosesdecoder script
        to split the symbols in the sentences to be space-delimited
        """
        utils.force_print("Tokenizing data... ")

        utils.make_dir("data/")
        for directory in [self.lang1_dir, self.lang2_dir]:
            new_data = utils.make_filename_from_filepath(directory)
            if utils.file_exists("data/" + new_data + ".tok"):
                continue
            command =  "/Users/urielmandujano/tools/mosesdecoder/scripts/" + \
                        "tokenizer/tokenizer.perl -q -threads " + \
                        "{} ".format(NUM_CPUS) + "< {}".format(directory) + \
                        " > {}".format("data/" + new_data + ".tok")
            subprocess.call(command, shell=True)

        utils.force_print("Done\n")

    def _load_tokenized_data(self):
        """
        Loads existing tokenized data into memory. Saves needless re-comp
        """
        new_class_vars = ['lang1_tokenized', 'lang2_tokenized']
        directories = [self.lang1_dir, self.lang2_dir]

        for var, d in zip(new_class_vars, directories):
            tok_data = utils.make_filename_from_filepath(d)
            parsed = self._parse_tokenized_data("data/" + tok_data + ".tok")
            setattr(self, var, parsed)

        utils.assert_equal_lens(self.lang1_tokenized, self.lang2_tokenized)

    def _parse_tokenized_data(self, tokdata_filename):
        """
        Given the tokenized data's filename and a counter, this function
        parses the data and returns it.
        """
        with open(tokdata_filename, 'r') as datafile:
            return datafile.read().split('\n')

    def _clean_corpus(self):
        """
        Lowercase the entire line, strip the line of non-ascii chars,
        drop empty lines, short lines, or long lines.
        """
        utils.force_print("Cleaning corpus... ")

        min_line_len = 0
        max_line_len = 100
        pop_indices = []
        for i, (l1, l2) in enumerate(zip(self.lang1_tokenized, \
                                         self.lang2_tokenized)):
            l1, l2 = l1.lower().split(), l2.lower().split()
            if l1 == l2 == []:
                pop_indices.append(i)
            elif len(l1) <= min_line_len or len(l2) <= min_line_len:
                pop_indices.append(i)
            elif len(l1) > max_line_len or len(l2) > max_line_len:
                pop_indices.append(i)
            else:
                self.lang1_tokenized[i] = self._strip_nonascii(' '.join(l1))
                self.lang2_tokenized[i] = self._strip_nonascii(' '.join(l2))

        for i in pop_indices[::-1]:
            self.lang1_tokenized.pop(i), self.lang2_tokenized.pop(i)

        self._save_cleansed_data()
        utils.force_print("Done\n")

    def _save_cleansed_data(self):
        """
        Saves newly cleansed data to data/ directory with the .cleansed
        extension name
        """
        utils.make_dir("data/")
        for directory, var in [[self.lang1_dir, self.lang1_tokenized],
                               [self.lang2_dir, self.lang2_tokenized]]:
            new_data = utils.make_filename_from_filepath(directory)
            datafile = "data/" + new_data + ".cleansed"
            utils.pickle_data(var, datafile)

    def _load_cleansed_data(self):
        """
        Loads existing cleansed data into memory
        """
        utils.force_print("Cleansed data found. Loading... ")

        new_class_vars = ['lang1_cleansed', 'lang2_cleansed']
        directories = [self.lang1_dir, self.lang2_dir]
        for var, d in zip(new_class_vars, directories):
            clean_data = utils.make_filename_from_filepath(d)
            parsed = utils.unpickle_data("data/" + clean_data + \
                                                ".cleansed")
            setattr(self, var, parsed)

        utils.assert_equal_lens(self.lang1_cleansed, self.lang2_cleansed)
        utils.force_print("Done\n")


    def get_vocab(self):
        """
        Creates a dictionary containing the vocabulary of each language.
        Keys are words, values are counts
        """
        try:
            return self.lang1_vocab, self.lang2_vocab
        except AttributeError:
            self.lang1_vocab, self.lang2_vocab = dict(), dict()
            self._create_lang_vocab(self.lang1_vocab, self.lang1_cleansed)
            self._create_lang_vocab(self.lang2_vocab, self.lang2_cleansed)
            return self.lang1_vocab, self.lang2_vocab

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
        try:
            return sorted(self.lang1_vocab.keys()), \
                   sorted(self.lang2_vocab.keys())
        except AttributeError:
            self.get_vocab()
            return sorted(self.lang1_vocab.keys()), \
                   sorted(self.lang2_vocab.keys())

    def get_corpus(self):
        """
        Returns the text data from both languages as a tuple of lists
        """
        return self.lang1_cleansed, self.lang2_cleansed

    def get_random_subset_corpus(self, percentage):
        """
        Returns a subset of the corpus data as a tuple of lists
        Ideal for low-memory settings
        """
        size = len(self.lang1_cleansed)
        subsize = int(percentage * len(self.lang1_cleansed))
        sample_indices = random.sample(range(0, size), subsize)

        lang1_data = [self.lang1_cleansed[i] for i in sample_indices]
        lang2_data = [self.lang2_cleansed[i] for i in sample_indices]

        del self.lang1_cleansed, self.lang2_cleansed
        self.lang1_cleansed = lang1_data
        self.lang2_cleansed = lang2_data
        return lang1_data, lang2_data

    def make_subset_vocab(self, lang1, lang2):
        """
        Given a subset of the languages, creates a vocabulary and returns
        the result as a dictionary whose key is a word and value is the
        count
        """
        lang1_vocab, lang2_vocab = dict(), dict()
        self._create_lang_vocab(lang1_vocab, lang1)
        self._create_lang_vocab(lang2_vocab, lang2)
        return lang1_vocab, lang2_vocab

    def _strip_nonascii(self, line):
        """
        Code to remove non-ascii characters from textfiles.
        Taken from jedwards on StackOverflow.
        """
        return line.decode('ascii', errors='ignore')

def main():
    lang1 = '/Users/urielmandujano/data/europarl/europarl-v7.es-en.en'
    lang2 = '/Users/urielmandujano/data/europarl/europarl-v7.es-en.es'
    euro_parser = EuroParlParser(lang1, lang2)

if __name__ == '__main__':
    main()
