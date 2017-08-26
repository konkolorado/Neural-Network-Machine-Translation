#python3.6

"""
Source code for training the translation system
"""
import os
import sys
import subprocess

import utilities

class Train(object):
    def __init__(self, path_to_moses, NCPUS, NGRAM, verbose=False):
        self.path_to_moses = path_to_moses
        self.NCPUS = NCPUS
        self.NGRAM = NGRAM
        self.lmdir = "lm/"
        utilities.make_dir(self.lmdir)
        self.verbose = verbose

    def _print(self, item):
        if self.verbose:
            utilities.flush_print(item)

    def _validate_file(self, src_file):
        """ Checks the provided files exist. Exits if theres an issue """
        if not os.path.exists(src_file):
            utilities.flush_print("Train:ERROR Invalid File: " + \
                src_file + "\n")
            sys.exit()

    def build_language_models(self, datafile):
        """
        Building the language model ensures fluid output. Should only
        be built for the target language in each pivot.
        In the Moses tutortial, the .lm file corresponds to the .arpa
        file
        """
        self._validate_file(datafile)
        lm_file = self.lmdir + utilities.strip_filename_from_path(datafile) + ".lm"
        blm_file = self.lmdir + utilities.strip_filename_from_path(datafile) + ".blm"
        if utilities.file_exists(lm_file) and utilities.file_exists(blm_file):
            return

        self._print("Building and binarizing language models... ")
        command = self.path_to_moses + "bin/lmplz "\
              "-o {} ".format(self.NGRAM) + \
              "--text " + datafile + \
              " --arpa " + lm_file + \
              " >> {} 2>&1".format(self.lmdir + "lm.out")
        subprocess.call(command, shell=True)
        self.binarize_language_model(lm_file, blm_file)
        self._print("Done\n")

    def binarize_language_model(self, lm_file, blm_file):
        """
        Binarizes the 2 target language model files for faster loading.
        Recommended for larger languages
        """
        command = self.path_to_moses + "bin/build_binary " + \
            lm_file + " " + blm_file + " >> {} 2>&1".format(self.lmdir + "blm.out")
        subprocess.call(command, shell=True)

    def train(self, src_file, tar_file, working_dir):
        """
        Carries out the training.  Creates a working directory,
        extracts the root file information and file extension information
        necessary for moses to run.  Sends output messages to working_dir/log
        """
        if utilities.dir_exists(working_dir):
            return

        self._validate_file(src_file)
        self._validate_file(tar_file)

        cwd = os.getcwd() + "/"
        blm = cwd + "lm/" + utilities.strip_filename_from_path(tar_file) + ".blm"

        shared = self._find_common_beginning(src_file, tar_file)
        file1_ext = src_file[shared+1:]
        file2_ext = tar_file[shared+1:]
        fileroot = cwd + src_file[:shared]
        log = "train.out"

        utilities.make_dir(working_dir)
        self._print("Training model at {}. This may take a while... ".format(working_dir))
        trainer = self.path_to_moses + "scripts/training/train-model.perl"
        command = "cd {};".format(working_dir) +\
            " nohup nice " + trainer + \
            " -root-dir train -corpus {}".format(fileroot) + \
            " -f {} -e {} -alignment".format(file1_ext, file2_ext) + \
            " grow-diag-final-and -reordering msd-bidirectional-fe" + \
            " -lm 0:3:{}:8".format(blm) + \
            " -cores {}".format(self.NCPUS) + \
            " -mgiza --parallel" + \
            " -external-bin-dir " + self.path_to_moses + "tools/mgizapp/" + \
            " >& {};".format(log) + \
            " cd .."
        subprocess.call(command, shell=True)
        self._print("Done\n")

    def _find_common_beginning(self, s1, s2):
        """ Given two strings, returns the index of the '.' character after which
        the first difference in strings occurs """
        index = 0
        while s1[index] == s2[index]:
            index += 1
        return s1[:index].rfind('.')

def main():
    config = utilities.config_file_reader()
    path_to_moses = utilities.safe_string(config.get("Environment Settings", "path_to_moses_decoder"))
    NGRAM = config.getint("Environment Settings", "ngram")
    NCPUS = config.getint("Environment Settings", "ncpus")

    trainer = Train(path_to_moses, NCPUS, NGRAM)
    trainer.build_language_models("data/train/europarl-v7.es-en.en.tok.cleansed.train")
    trainer.build_language_models("data/train/europarl-v7.fr-en.fr.tok.cleansed.train")

    trainer.train("data/train/europarl-v7.es-en.es.tok.cleansed.train",
        "data/train/europarl-v7.es-en.en.tok.cleansed.train", "es-en.working")
    trainer.train("data/train/europarl-v7.fr-en.en.tok.cleansed.train",
        "data/train/europarl-v7.fr-en.fr.tok.cleansed.train", "en-fr.working")


if __name__ == '__main__':
    main()
