"""
Source code for tuning the translation system
"""
import os
import sys
import subprocess

import utilities

class Tune(object):

    def __init__(self, path_to_moses, NCPUS, verbose=False):
        self.path_to_moses = path_to_moses
        self.NCPUS = NCPUS
        self.verbose = verbose

    def _print(self, item):
        if self.verbose:
            utilities.flush_print(item)

    def _validate_file(self, src_file):
        """ Checks the provided files exist. Exits if theres an issue """
        if not os.path.exists(src_file):
            utilities.flush_print("Tune:ERROR Invalid File: " + \
                src_file + "\n")
            sys.exit()

    def tune(self, src_tune, tar_tune, working_dir):
        """
        Initiates the tuning routine. This will take a while. Changes
        working directory into the appropriate directory, executes the
        command and returns to the project's base directory.
        """
        if utilities.dir_exists(working_dir + "/mert-work") and \
            utilities.file_exists(working_dir + "/mert-work/moses.ini"):
            return

        if not utilities.isabsolute(src_tune):
            src_tune = os.getcwd() + "/" + src_tune
        if not utilities.isabsolute(tar_tune):
            tar_tune = os.getcwd() + "/" + tar_tune
        self._validate_file(src_tune)
        self._validate_file(tar_tune)

        self._print("Tuning model at {}. This may take a while... ".format(working_dir))
        command = "cd {};".format(working_dir) + \
            "nohup nice " + \
            self.path_to_moses + "scripts/training/mert-moses.pl" + \
            " {} {} ".format(src_tune, tar_tune) + \
            self.path_to_moses + "bin/moses train/model/moses.ini" + \
            " --mertdir " + self.path_to_moses + "bin/" + \
            ' --decoder-flags="-threads {}"'.format(self.NCPUS) + \
            " &> mert.out; cd .."
        subprocess.call(command, shell=True)
        self._print("Done\n")


def main():
    config = utilities.config_file_reader()
    NCPUS = config.getint("Environment Settings", "ncpus")
    path_to_moses = utilities.safe_string(config.get("Environment Settings", "path_to_moses_decoder"))

    tuner = Tune(path_to_moses, NCPUS)
    tuner.tune("data/tune/europarl-v7.es-en.es.tok.cleansed.tune",
        "data/tune/europarl-v7.es-en.en.tok.cleansed.tune", "es-en.working")
    tuner.tune("data/tune/europarl-v7.fr-en.en.tok.cleansed.tune",
        "data/tune/europarl-v7.fr-en.fr.tok.cleansed.tune", "en-fr.working")

if __name__ == '__main__':
    main()
