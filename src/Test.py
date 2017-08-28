"""
Source code for testing the translation system
"""
import os
import sys
import subprocess

import psutil

import utilities

class Test(object):
    def __init__(self, path_to_moses, verbose = False):
        self.path_to_moses = path_to_moses
        self.verbose = verbose

    def _print(self, item):
        if self.verbose:
            utilities.flush_print(item)

    def _validate_file(self, src_file):
        """ Checks the provided files exist. Exits if theres an issue """
        if not os.path.exists(src_file):
            utilities.flush_print("Test:ERROR Invalid File: " + \
                src_file + "\n")
            sys.exit()

    def test_translation_quality(self, src_test, tar_test, working_dir):
        """
        Given two files containing source data and target data, outputs
        the bleu score for translation.  A higher bleu means better
        translation quality.
        """
        self._validate_file(src_test), self._validate_file(tar_test)
        assert utilities.dir_exists(working_dir), "TestTranslationQualityError: {} not found".format(working_dir)

        if not utilities.isabsolute(src_test):
            src_test = os.getcwd() + "/" + src_test

        # Filter test set
        filt_dir = working_dir + "/binarized-filtered-model"
        if not utilities.dir_exists(filt_dir):
            self._filter_test_set(src_test, working_dir, filt_dir, "binarizer.out")

        # Create translated file
        result = src_test + ".translated"
        if not utilities.file_exists(result):
            self._translate_pivot(src_test, working_dir, result, filt_dir, "translation.out")

        # Find bleu score
        result_file = working_dir + "/translation.bleu"
        if not utilities.file_exists(result_file):
            self._get_bleu_score(src_test + ".translated", tar_test, working_dir, result_file)

        # Report bleu score
        self._report_bleu_score(working_dir, result_file)

    def _filter_test_set(self, src_test, working_dir, filt_dir, debug):
        """
        Filter the trained model so that we retain only the entries
        necessary to translate the test set.  Makes the final
        translation much faster
        """
        self._print("Filtering test set at {}... ".format(working_dir))
        command = self.path_to_moses + "scripts/" + \
            "training/filter-model-given-input.pl" + \
            " {} {}/mert-work/moses.ini".format(filt_dir, working_dir) + \
            " {}".format(src_test) + \
            " -Binarizer " + self.path_to_moses + "bin/processPhraseTableMin" + \
            " &> {}/{}".format(working_dir, debug)
        subprocess.call(command, shell=True)
        self._print("Done\n")

    def _translate_pivot(self, src_test, working_dir, result, filt_dir, debug):
        """
        Given two files containing src and target data, returns the bleu score
        """
        self._print("Translating between langs in {}.\n\tSaving to {}... ".format(working_dir, result))

        command ="nohup nice " + \
            self.path_to_moses + "bin/moses"+\
            " -f {}/moses.ini <".format(filt_dir) + \
            " {}".format(src_test) + \
            " > {}".format(result) + \
            " 2> {}/{}".format(working_dir, debug) + \
            " -minlexr-memory"
        subprocess.call(command, shell=True)
        self._print("Done\n")

    def _get_bleu_score(self, src_translated, tar_test, working_dir, result_file):
        """
        Runs the moses script to eval bleu scores on a completed translation
        """
        self._print("Obtaining bleu scores for {}... ".format(src_translated))
        command = self.path_to_moses + "scripts/" + \
                  "generic/multi-bleu.perl -lc {}".format(tar_test) + \
                  " < {} > {}".format(src_translated, result_file)
        subprocess.call(command, shell=True)
        self._print("Done\n")

    def _report_bleu_score(self, working_dir, result_file):
        """
        Opens the file containing the bleu score and reports it to the user
        """
        assert utilities.file_exists(result_file), "Error {} not found".format(result_file)
        print("Results for {} translation".format(working_dir))
        print("\t", open(result_file, 'r').readline().strip(), "\n")

    def test_pivoting_quality(self, src_test, src_working_dir, tar_test, tar_working_dir):
        """
        Tests the quality of translation via a pivoting language.
        Returns the bleu score to the user. As parameters, this expects
        the src language test file,
        the src language to piv directory containing the trained decoder,
        the target language test file,
        and the piv to tar language directory containing the trained
        decoder
        """
        self._validate_file(src_test), self._validate_file(tar_test)
        assert utilities.dir_exists(src_working_dir), "TestPivotingQualityError: {} not found".format(src_working_dir)
        assert utilities.dir_exists(tar_working_dir), "TestPivotingQualityError: {} not found".format(tar_working_dir)

        if not utilities.isabsolute(src_test):
            src_test = os.getcwd() + "/" + src_test

        filt_dir = src_working_dir + "/pivot-binarized-filtered-model/"
        if not utilities.dir_exists(filt_dir):
            debug = "pivot.binarizer.out"
            self._filter_test_set(src_test, src_working_dir, filt_dir, debug)

        trans_result = src_test + ".pivot.translated"
        if not utilities.file_exists(trans_result):
            debug = "pivot.translation.out"
            self._translate_pivot(src_test, src_working_dir, trans_result, filt_dir, debug)

        tar_filt_dir = tar_working_dir + "/pivot-binarized-filtered-model/"
        if not utilities.dir_exists(tar_filt_dir):
            debug = "pivot.binarizer.out"
            self._filter_test_set(trans_result, tar_working_dir, tar_filt_dir, debug)

        target_result = trans_result + ".final"
        if not utilities.file_exists(target_result):
            debug = "pivot.translation.out"
            self._translate_pivot(trans_result, tar_working_dir, target_result, tar_filt_dir, debug)

        result_file = tar_working_dir + "/pivot.translation.bleu"
        if not utilities.file_exists(result_file):
            self._get_bleu_score(target_result, tar_test, tar_working_dir, result_file)

        self._report_bleu_score("pivot", result_file)

    def translate_file(self, src_test, working_dir):
        """
        Translates a file from one language into another.  This relies on having a
        properly trained and tuned system for the target languages available in the
        working_dir.
        As parameters, this expects the src_file to translate and the working_dir
        containing the trained translation system
        """
        self._validate_file(src_test)
        assert utilities.dir_exists(working_dir), "TestTranslationQualityError: {} not found".format(working_dir)

        if not utilities.isabsolute(src_test):
            src_test = os.getcwd() + "/" + src_test

        # Filter test set
        filt_dir = working_dir + "/binarized-filtered-model"
        if not utilities.dir_exists(filt_dir):
            self._filter_test_set(src_test, working_dir, filt_dir, "binarizer.out")

        # Create translated file
        result = src_test + ".translated"
        if not utilities.file_exists(result):
            self._translate_pivot(src_test, working_dir, result, filt_dir, "translation.out")
        else:
            print("Saved {} translation result to {}".format(src_test, result))

def main():
    config = utilities.config_file_reader()
    path_to_moses = utilities.safe_string(config.get("Environment Settings", "path_to_moses_decoder"))

    test = Test(path_to_moses)
    test.test_translation_quality("data/test/europarl-v7.es-en.es.tok.cleansed.test",
        "data/test/europarl-v7.es-en.en.tok.cleansed.test", "es-en.working")
    test.test_translation_quality("data/test/europarl-v7.fr-en.en.tok.cleansed.test",
        "data/test/europarl-v7.fr-en.fr.tok.cleansed.test", "en-fr.working")
    test.test_pivoting_quality("data/test/europarl-v7.es-en.es.tok.cleansed.test.matched",
        "es-en.working", "data/test/europarl-v7.fr-en.fr.tok.cleansed.test.matched", "en-fr.working")

if __name__ == '__main__':
    main()
