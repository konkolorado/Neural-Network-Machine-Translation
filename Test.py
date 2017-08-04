#python3.6

"""
Source code for testing the translation system
"""
import os
import sys
import subprocess
import socket
import xmlrpc.client
from multiprocessing import Process

import psutil

import utilities

config = utilities.config_file_reader()
path_to_moses_decoder = utilities.safe_string(config.get("Environment Settings", "path_to_moses_decoder"))

class Test(object):
    def __init__(self, verbose = False):
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

    def test_translator_interactive(self, working_dir):
        """
        Launches the mosesdecoder to allow for interactive decoding between
        source and target languages.
        """
        assert utilities.dir_exists(working_dir), "TestInteractiveError: {} not found".format(working_dir)

        temp_file = working_dir + "/interactive.out"
        port = self._get_free_port()
        proxy = self._setup_proxy(port)

        process = self._load_server(working_dir, port, temp_file)
        self._manage_connections(process, proxy)
        self._shut_server(process)

    def _load_server(self, working_dir, port, logfile):
        """
        Loads the moses server on the provided port number using the
        informatio in the specified working directory. Returns the
        launched server process
        """
        self._print("Loading interactive translator at {}...".format(working_dir))
        command = path_to_moses_decoder + "bin/moses" + \
            " -minlexr-memory --server --server-port {}".format(port) + \
            " --server-maxconn-backlog 5" + \
            " -v 0 -f {}/mert-work/moses.ini &".format(working_dir)

        with open(logfile, 'w') as err:
            process = subprocess.Popen(command.split(), shell=False, stderr=err)
        self._print("Ready\n")
        return process

    def _make_translation_request(self, proxy, text):
        """ Sends the text we want to translate to the moses server """
        response = proxy.translate({"text": text})
        return response["text"]

    def _get_free_port(self):
        """ Returns an available port number for moses server to use """
        sock = socket.socket()
        sock.bind(('', 0))
        return sock.getsockname()[1]

    def _setup_proxy(self, port):
        """ Sets up a server proxy to communicate with moses server """
        return xmlrpc.client.ServerProxy("http://localhost:{}/RPC2".format(port))

    def _shut_server(self, process):
        """ Uses the putil library to shut the background translation
        service down """
        psutil.Process(process.pid).kill()

    def _manage_connections(self, process, proxy):
        """ Accepts user input, submits it to moses, returns the result """
        print("Enter text to translate (type quit to exit)")
        while True:
            query = input(">> ").lower().strip()
            if query == "quit" or query == "q":
                return

            try:
                result = self._make_translation_request(proxy, query)
            except (ConnectionRefusedError, xmlrpc.client.Fault) as e:
                result = ''

            print("Text: {}\tTranslation: {}\n".format(query, result))

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
        command = path_to_moses_decoder + "scripts/" + \
            "training/filter-model-given-input.pl" + \
            " {} {}/mert-work/moses.ini".format(filt_dir, working_dir) + \
            " {}".format(src_test) + \
            " -Binarizer " + path_to_moses_decoder + "bin/processPhraseTableMin" + \
            " &> {}/{}".format(working_dir, debug)
        subprocess.call(command, shell=True)
        self._print("Done\n")

    def _translate_pivot(self, src_test, working_dir, result, filt_dir, debug):
        """
        Given two files containing src and target data, returns the bleu score
        """
        self._print("Translating between langs in {}.\n\tSaving to {}... ".format(working_dir, result))
        command ="nohup nice " + \
            path_to_moses_decoder + "bin/moses"+\
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
        command = path_to_moses_decoder + "scripts/" + \
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

    def test_pivoting_interactive(self, working_dir1, working_dir2):
        """
        Launches the mosesdecoder to allow for interactive pivoting decoding
        from the source language into the pivot language and on to the
        target language
        """
        assert utilities.dir_exists(working_dir1), "TestInteractiveError: {} not found".format(working_dir1)
        assert utilities.dir_exists(working_dir2), "TestInteractiveError: {} not found".format(working_dir2)

        temp_file1 = working_dir1 + "/interactive.out"
        temp_file2 = working_dir2 + "/interactive.out"

        port1, port2 = self._get_free_port(), self._get_free_port()
        prox1, prox2 = self._setup_proxy(port1), self._setup_proxy(port2)

        process1 = self._load_server(working_dir1, port1, temp_file1)
        process2  =self._load_server(working_dir2, port2, temp_file2)

        self._manage_pivoting_connections(prox1, prox2)

        self._shut_server(process1)
        self._shut_server(process2)

    def _manage_pivoting_connections(self, prox1, prox2):
        """
        Accepts user input, submits it to moses for the pivoting
        translation and displays the result to the user
        """
        print("Enter text to translate (type quit to exit)")
        while True:
            query = input(">> ").lower().strip()
            if query == "quit" or query == "q":
                return
            try:
                piv_result = self._make_translation_request(prox1, query)
                tar_result = self._make_translation_request(prox2, piv_result)
            except (ConnectionRefusedError, xmlrpc.client.Fault) as e:
                tar_result = ''

            print("Text: {}\tTranslation: {}\n".format(query, tar_result))

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
