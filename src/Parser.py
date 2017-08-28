"""
Author: Uriel Mandujano
Europarl data available at
    http://www.statmt.org/europarl/

For use with the Moses decoder
NOTE: Moses decoder only compatible with:
    - all words must be lowercased
    - 1 sentence per line, no empty lines
    - sentences no longer than 100 words

- pip3 install pympler
"""

import os
import sys
import subprocess
import random
from collections import deque
import numpy
from itertools import zip_longest
import hashlib
import linecache
import operator

from pympler import asizeof

import utilities

class Parser(object):
    def __init__(self, path_to_moses, mem_limit, max_len, min_len, verbose=False):
        self.path_to_moses = path_to_moses
        self.mem_limit = mem_limit
        self.max_len = max_len
        self.min_len = min_len

        self.destdir = "data/"
        self.traindir = self.destdir + "train/"
        self.tunedir = self.destdir + "tune/"
        self.testdir = self.destdir + "test/"
        self.verbose = verbose
        utilities.make_dir(self.destdir)

    def _print(self, item):
        if self.verbose:
            utilities.flush_print(item)

    def _validate_file(self, src_file):
        """ Checks the provided files exist. Exits if theres an issue """
        if not os.path.exists(src_file):
            utilities.flush_print("EuroParlParser:ERROR Invalid File: " + \
                src_file + "\n")
            sys.exit()

    def tokenize_files(self, files):
        for f in files:
            self.tokenize(f)

    def tokenize(self, src_file):
        """
        Tokenize the file provided using the mosesdecoder script
        by splitting the symbols in the sentences to be space-delimited
        """
        self._validate_file(src_file)
        dest_file = self.destdir + utilities.strip_filename_from_path(src_file) + ".tok"
        if utilities.file_exists(dest_file):
            return

        self._print("""Running tokenizer. """
            """Splitting into space delimited tokens... """)
        command = self.path_to_moses + "scripts/tokenizer/tokenizer.perl " + \
            "-q -threads {} ".format(NUM_CPUS) + \
            "< {}".format(src_file) + \
            " > {}".format(dest_file)
        subprocess.call(command, shell=True)
        self._print("Done\n")

    def cleanse(self, src_lang_file, tar_lang_file):
        """
        Cleans the file provided by lowercasing all words and ensuring each line in
        the text file is within min_len and max_len. Operates on two streams
        simultaneously in order to keep line to line correspondence
        """
        self._validate_file(src_lang_file), self._validate_file(tar_lang_file)
        src_dest_file = self.destdir + utilities.strip_filename_from_path(src_lang_file) + ".cleansed"
        tar_dest_file = self.destdir + utilities.strip_filename_from_path(tar_lang_file) + ".cleansed"

        if utilities.files_exist([src_dest_file, tar_dest_file]):
            return
        else:
            utilities.wipe_files([src_dest_file, tar_dest_file])
        self._print("""Cleaning data.  Ensuring uniformity of data...""")

        src_buf, tar_buf = [], []
        for src_line, tar_line in zip(open(src_lang_file), open(tar_lang_file)):
            src_line = src_line.lower().split()
            tar_line = tar_line.lower().split()

            if len(src_line) > self.min_len and len(src_line) < self.max_len and \
                len(tar_line) > self.min_len and len(tar_line) < self.max_len:
                src_buf.append(' '.join(src_line))
                tar_buf.append(' '.join(tar_line))

            if asizeof.asizeof(src_buf) + asizeof.asizeof(tar_buf) > self.mem_limit:
                self._dump_bufs_to( [src_dest_file, tar_dest_file],
                                    [src_buf, tar_buf])

        self._dump_bufs_to([src_dest_file, tar_dest_file], [src_buf, tar_buf])
        self._print("Done\n")

    def _dump_buf_to(self, filename, ls):
        """ Opens a file and appends a lists contents to it """
        outstream = open(filename, 'a')
        for item in ls:
            outstream.write("{}\n".format(item))
        outstream.close()
        ls[:] = []

    def _dump_bufs_to(self, files, datas):
        """
        Takes two lists: the first is a list of filenames, the second
        is a list of lists containing the data we want to dump.
        The filenames and list's order must correspond.
        """
        for f, d in zip(files, datas):
            self._dump_buf_to(f, d)

    def subset(self, src_lang_file, tar_lang_file, proportion, subdir = ""):
        """ Creates a new proportion of data set to create new datasets.
        Maintains the correspondence of entries between two data files.
        Takes as parameters the two files that must be subset, the fraction
        of data that should be taken as a subset (1/3), and an option subdir
        directory where the files should be placed within the data dir """
        self._validate_file(src_lang_file), self._validate_file(tar_lang_file)

        src_dest_file = self.destdir + subdir + utilities.strip_filename_from_path(src_lang_file) + ".subset"
        tar_dest_file = self.destdir + subdir + utilities.strip_filename_from_path(tar_lang_file) + ".subset"

        if utilities.files_exist([src_dest_file, tar_dest_file]):
            return
        else:
            utilities.wipe_files([src_dest_file, tar_dest_file])

        self._print("""Choosing a random subset of the data...""")
        text_size = self._min_text_size(src_lang_file, tar_lang_file)
        subset_size = int(proportion * text_size)
        assert subset_size > 0, "Subset length must be non-zero"

        subset_lines = deque(sorted(random.sample(range(0, text_size), subset_size)))
        self._get_lines(src_lang_file, tar_lang_file, subset_lines, src_dest_file, tar_dest_file)
        self._print("Done\n")

    def _min_text_size(self, text1, text2):
        """ Opens 2 files and returns the number of lines in the smallest """
        return min(sum(1 for line in open(text1)), \
            sum(1 for line in open(text2)))

    def split_train_tune_test(self, src_file, src_piv_file, piv_tar_file, tar_file,
        train_split, test_split):
        """
        Splits the full datafiles into test, tune, and train sets.
        Receives 4 files as parameters and 2 decimals indicating the percentage of
        data to be used as train, tune, and test data. If line 1 in src langs is
        in test, then line 1 in tar langs will also be in test. Etc.
        """
        utilities.make_dir(self.traindir)
        utilities.make_dir(self.tunedir)
        utilities.make_dir(self.testdir)

        self._validate_file(src_file), self._validate_file(src_piv_file)
        self._validate_file(piv_tar_file), self._validate_file(tar_file)
        assert train_split + test_split <= 1 , "Invalid size for train, tune, and test splits"

        train_files, tune_files, test_files = self._ttt_filenames(src_file, src_piv_file, piv_tar_file, tar_file)
        if utilities.ttt_files_exist(train_files, tune_files, test_files):
            return
        else:
            utilities.ttt_wipe_files(train_files, tune_files, test_files)

        self._print("""Splitting data into train, tune, and test sets...""")
        train, tune, test = [[] ,[], [], []],  [[], [], [], []], [[], [], [], []]
        for src_line, src_piv_line, piv_tar_line, tar_line in \
            zip_longest(open(src_file), open(src_piv_file), open(piv_tar_file), open(tar_file)):

            x = numpy.random.sample()
            if x < train_split:
                self._add_line_to(train[0], src_line)
                self._add_line_to(train[1], src_piv_line)
                self._add_line_to(train[2], piv_tar_line)
                self._add_line_to(train[3], tar_line)
            elif x >= train_split and x < train_split + test_split:
                self._add_line_to(tune[0], src_line)
                self._add_line_to(tune[1], src_piv_line)
                self._add_line_to(tune[2], piv_tar_line)
                self._add_line_to(tune[3], tar_line)
            else:
                self._add_line_to(test[0], src_line)
                self._add_line_to(test[1], src_piv_line)
                self._add_line_to(test[2], piv_tar_line)
                self._add_line_to(test[3], tar_line)

            if asizeof.asizeof(train) + asizeof.asizeof(tune) + \
                asizeof.asizeof(test) > self.mem_limit:
                self._dump_ttt_bufs_to(train, tune, test, train_files, tune_files, test_files)

        self._dump_ttt_bufs_to(train, tune, test, train_files, tune_files, test_files)
        self._print("Done\n")

    def _add_line_to(self, ls, line):
        """
        Adds the contents of line to list ls after checking if
        the line is not None and stripping it
        """
        if line != None:
            ls.append(line.strip())

    def _dump_ttt_bufs_to(self, train, tune, test, train_files, tune_files, test_files):
        """
        Dumps the train, tune, test data into their corresponding files.
        Empties the data.
        """
        self._dump_bufs_to(train_files, train)
        self._dump_bufs_to(tune_files, tune)
        self._dump_bufs_to(test_files, test)
        train[:] = [[], [], [], []]
        test[:] = [[], [], [], []]
        tune[:] = [[], [], [], []]

    def _ttt_filenames(self, src_file, src_piv_file, piv_tar_file, tar_file):
        """
        Constructs the appropriate train tune test file extension names for the data.
        Returns a list of lists, where the list in index 0 is the name of the train
        files, the list in index 1 is the tune files, and index 2 is the test
        """
        src_train_file = self.traindir + utilities.strip_filename_from_path(src_file) + ".train"
        src_tune_file = self.tunedir + utilities.strip_filename_from_path(src_file) + ".tune"
        src_test_file = self.testdir + utilities.strip_filename_from_path(src_file) + ".test"

        src_piv_train_file = self.traindir + utilities.strip_filename_from_path(src_piv_file) + ".train"
        src_piv_tune_file = self.tunedir + utilities.strip_filename_from_path(src_piv_file) + ".tune"
        src_piv_test_file = self.testdir + utilities.strip_filename_from_path(src_piv_file) + ".test"

        piv_tar_train_file = self.traindir + utilities.strip_filename_from_path(piv_tar_file) + ".train"
        piv_tar_tune_file = self.tunedir + utilities.strip_filename_from_path(piv_tar_file) + ".tune"
        piv_tar_test_file = self.testdir + utilities.strip_filename_from_path(piv_tar_file) + ".test"

        tar_train_file = self.traindir + utilities.strip_filename_from_path(tar_file) + ".train"
        tar_tune_file = self.tunedir + utilities.strip_filename_from_path(tar_file) + ".tune"
        tar_test_file = self.testdir + utilities.strip_filename_from_path(tar_file) + ".test"

        train_files = [src_train_file, src_piv_train_file, piv_tar_train_file, tar_train_file]
        tune_files = [src_tune_file, src_piv_tune_file, piv_tar_tune_file, tar_tune_file]
        test_files = [src_test_file, src_piv_test_file, piv_tar_test_file, tar_test_file]
        return train_files, tune_files, test_files

    def match(self, src_file, src_piv_file, piv_tar_file, tar_file):
        """
        Opens src_piv_file and piv_tar_file and keeps track of
        only those sentences which occur in both.  Then, it
        deletes all lines from the files except for the lines
        which contain a match.  This step is necessary for the
        translation via pivotting to make sure that testing is
        accurate
        """
        self._validate_file(src_file), self._validate_file(src_piv_file)
        self._validate_file(piv_tar_file), self._validate_file(tar_file)

        m_src_file = src_file + ".matched"
        m_src_piv_file = src_piv_file + ".matched"
        m_piv_tar_file = piv_tar_file + ".matched"
        m_tar_file = tar_file + ".matched"

        if utilities.files_exist([m_src_file, m_src_piv_file, m_piv_tar_file, m_tar_file]):
            return
        else:
            utilities.wipe_files([m_src_file, m_src_piv_file, m_piv_tar_file, m_tar_file])

        self._print("Starting matching... ")
        hash_to_index = self._make_hash_to_index(src_piv_file)
        index_to_index = self._make_index_to_index(piv_tar_file, hash_to_index)

        self._get_relevant_lines_in_first_pivot(src_file, src_piv_file, index_to_index,
            m_src_file, m_src_piv_file)

        self._get_relevant_lines_in_second_pivot(piv_tar_file, tar_file, index_to_index,
            m_piv_tar_file, m_tar_file)

        self._print("Done\n")

    def _make_hash_to_index(self, filename):
        """
        Opens a file and creates a dict which stores each line's hash
        as a key, and the line number as its value. Returns dict
        """
        hash_to_index = {}
        for i, line in enumerate(open(filename, 'r')):
            digest = hashlib.sha224(str.encode(line.strip())).hexdigest()
            hash_to_index[digest] = i
        return hash_to_index

    def _make_index_to_index(self, filename, hash_to_index):
        """
        Opens file filename, hashes each line one by one.  If a hash matches
        another hash in hash_to_index, then we know that line is a match
        with the hash's value in hash_to_index. We create a dictionary
        mapping from one files indices to the others
        """
        index_to_index = {}
        for i, line in enumerate(open(filename, 'r')):
            digest = hashlib.sha224(str.encode(line.strip())).hexdigest()
            if digest in hash_to_index:
                index_to_index[hash_to_index[digest]] =  i
        return index_to_index

    def _get_relevant_lines_in_first_pivot(self, src_file, src_piv_file, index_to_index,
        m_src_file, m_src_piv_file):
        """
        Given a dictionary mapping of relevant indices in one pivot to
        relevant indices in another, creates new files with the "matched"
        extension indicating that they contain only the lines shared
        within the texts in the common language. Saves the lines to m_file
        """
        m_src_buf, m_src_piv_buf = [], []
        first_piv_indices = deque(sorted(index_to_index))
        self._get_lines(src_file, src_piv_file, first_piv_indices, m_src_file, m_src_piv_file)

    def _get_relevant_lines_in_second_pivot(self, piv_tar_file, tar_file, index_to_index,
        m_piv_tar_file, m_tar_file):
        """
        Given a dictionary mapping of relevant indices in one pivot to
        relevant indices in another, creates new files with the "matched"
        extension indicating that they contain only the lines shared
        within the texts in the common language. Saves the lines to m_file
        """
        second_piv_indices = sorted(index_to_index.items(), key=operator.itemgetter(0))
        second_piv_indices = [x[1] for x in second_piv_indices]

        while True:
            inc_len = self._longest_non_decreasing_sequence(second_piv_indices, 0)
            target_indices =  second_piv_indices[:inc_len]
            self._get_lines(piv_tar_file, tar_file, target_indices, m_piv_tar_file, m_tar_file)
            second_piv_indices = second_piv_indices[inc_len:]
            if len(second_piv_indices) == 0:
                break

    def _longest_non_decreasing_sequence(self, arr, pos):
        """
        Given an arr and a starting pos, returns the length of non
        decreasing subsequence.
        """
        length = 1
        if len(arr) <= 1:
            return length

        while arr[pos] < arr[pos+1] :
            length += 1
            pos += 1
            if pos == len(arr) - 1:
                break
        return length

    def _get_lines(self, file1, file2, lines, file1_dest, file2_dest):
        """
        Given two files to open and a sorted list of lines to get, open the files,
        retrieves the desired lines, and dumps them to specified file locations.
        """
        lines = deque(lines)
        buf1, buf2 = [], []
        line_counter, target_line = 0, lines.popleft()

        for f1_line, f2_line in zip(open(file1, 'r'), open(file2, 'r')):
            if target_line == line_counter:
                buf1.append(f1_line.strip())
                buf2.append(f2_line.strip())

                if asizeof.asizeof(buf1) + asizeof.asizeof(buf2) > \
                    self.mem_limit:
                    self._dump_bufs_to( [file1_dest, file2_dest],
                                        [buf1, buf2])

                if len(lines) != 0:
                    target_line = lines.popleft()
                else:
                    break
            line_counter += 1

        self._dump_bufs_to( [file1_dest, file2_dest],
                            [buf1, buf2])

def main():
    config = utilities.config_file_reader()
    path_to_moses = utilities.safe_string(config.get("Environment Settings", "path_to_moses_decoder"))
    mem_limit = config.getint("Environment Settings", "mem_limit")
    max_len = config.getint("Iteration Settings", "max_sentence_len")
    min_len = config.getint("Iteration Settings", "min_sentence_len")
    parser = Parser(path_to_moses, mem_limit, max_len, min_len, True)

    parser.tokenize("data/src/europarl-v7.es-en.es")
    parser.tokenize("data/src/europarl-v7.es-en.en")
    parser.tokenize("data/src/europarl-v7.fr-en.en")
    parser.tokenize("data/src/europarl-v7.fr-en.fr")

    parser.cleanse("data/europarl-v7.es-en.es.tok", "data/europarl-v7.es-en.en.tok")
    parser.cleanse("data/europarl-v7.fr-en.en.tok", "data/europarl-v7.fr-en.fr.tok")

    parser.split_train_tune_test("data/europarl-v7.es-en.es.tok.cleansed", "data/europarl-v7.es-en.en.tok.cleansed",
        "data/europarl-v7.fr-en.en.tok.cleansed", "data/europarl-v7.fr-en.fr.tok.cleansed", .6, .2)

    # Makes sense to do this to the training/tune data if the training/tune data is too large
    parser.subset("data/train/europarl-v7.es-en.es.tok.cleansed.train", "data/train/europarl-v7.es-en.en.tok.cleansed.train", .5, "train/")
    parser.subset("data/train/europarl-v7.fr-en.en.tok.cleansed.train", "data/train/europarl-v7.fr-en.fr.tok.cleansed.train", .5, "train/")

    # Necessary to do for test data to be consistent
    parser.match("data/test/europarl-v7.es-en.es.tok.cleansed.test", "data/test/europarl-v7.es-en.en.tok.cleansed.test",
        "data/test/europarl-v7.fr-en.en.tok.cleansed.test", "data/test/europarl-v7.fr-en.fr.tok.cleansed.test")

if __name__ == '__main__':
    main()
