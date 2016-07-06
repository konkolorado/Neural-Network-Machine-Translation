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

"""
import os
import sys
import subprocess

NUM_CPUS = 4

class EuroParlParser(object):
    def __init__(self, lang1_dir, lang2_dir):
        self.lang1_dir = lang1_dir
        self.lang2_dir = lang2_dir
        self._check_lang_files_exist()

        if self._no_tokenized_data_exists():
            self._force_print("No tokenized data found. Tokenizing...")
            self._tokenize()
        self._load_tokenized_data()



    def __str__(self):
        return "{}\n{}".format(self.lang1_dir, self.lang2_dir)

    def _check_lang_files_exist(self):
        assert os.path.exists(self.lang1_dir), \
            "{} file not found".format(self.lang1_dir)
        assert os.path.exists(self.lang2_dir), \
            "{} file not found".format(self.lang2_dir)

    def _no_tokenized_data_exists(self):
        """
        Determines if tokenized data exists. If so, this function does nothing.
        If not, this function proceeds to tokenize the data
        """
        tokdat1 = self._make_filename_from_filepath(self.lang1_dir)
        tokdat2 = self._make_filename_from_filepath(self.lang2_dir)
        if os.path.exists('data/{}.tok'.format(tokdat1)) and \
           os.path.exists('data/{}.tok'.format(tokdat2)):
            return False
        return True

    def _tokenize(self):
        """
        Given a list of sentences, we tokenize using the mosesdecoder script
        to split the symbols in the sentences to be space-delimited
        """
        self._make_dir("data/")
        self.raw_data = []

        for directory in [self.lang1_dir, self.lang2_dir]:
            new_data = self._make_filename_from_filepath(directory)
            command =  "/Users/urielmandujano/tools/mosesdecoder/scripts/" + \
                        "tokenizer/tokenizer.perl -q -threads " + \
                        "{} ".format(NUM_CPUS) + "< {}".format(directory) + \
                        " > {}".format("data/" + new_data + ".tok")
            subprocess.call(command, shell=True)

    def _load_tokenized_data(self):
        """
        Loads existing tokenized data into memory. Saves needless re-comp
        """
        new_class_vars = ['lang1_tokenized', 'lang2_tokenized']
        directories = [self.lang1_dir, self.lang2_dir]

        for var, d in zip(new_class_vars, directories):
            tok_data = self._make_filename_from_filepath(d)
            parsed = self._parse_tokenized_data("data/" + tok_data + ".tok")
            setattr(self, var, parsed)

        self._assert_equal_lens(self.lang1_tokenized, self.lang2_tokenized)

    def _parse_tokenized_data(self, tokdata_filename):
        """
        Given the tokenized data's filename and a counter, this function
        parses the data and returns it.
        """
        with open(tokdata_filename, 'r') as datafile:
            return datafile.read().split('\n')

    def clean_corpus(self):
        """
        NOT WORKING
        Drop lines that are empty, too short, or too long.
        """
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
        """

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

    def _strip_nonascii(self, line):
        """
        Code to remove non-ascii characters from textfiles.
        Taken from jedwards on StackOverflow.
        """
        return line.decode('ascii', errors='ignore')

    def _assert_equal_lens(self, item1, item2):
        """
        Given two container items, assert that they contain an equal
        number of items and are non empty
        """
        assert len(item1) == len(item2), "Unequal language sizes"
        assert len(item1) and len(item2), "Got language of size 0"

    def _force_print(self, item):
        """
        Force prints the item to stdout
        """
        print item
        sys.stdout.flush()

    def _make_dir(self, path):
        """
        Determines if a given path name is a valid directory. If not, makes it
        """
        if not os.path.isdir(path):
            os.makedirs(path)

    def _make_filename_from_filepath(self, path):
        """
        Given a path to a file, this function finds the filename and returns
        it
        """
        return os.path.split(path)[1]

def main():
    lang1 = '/Users/urielmandujano/data/europarl/europarl-v7.es-en.en'
    lang2 = '/Users/urielmandujano/data/europarl/europarl-v7.es-en.es'
    euro_parser = EuroParlParser(lang1, lang2)

if __name__ == '__main__':
    main()
