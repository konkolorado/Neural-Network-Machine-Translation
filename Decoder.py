"""
Author: Uriel Mandujano
Decoder for use with Moses and the Eurparl data. Performs statistical
MT between two pairs of languages and by pivoting.

Requires Giza++ and KenLM, freely available software packages
-- Installing Giza++ [MGIZA] on OSX 10.11
install boost from website to /usr/local
cd ~/to/where/moses/is/
git clone https://github.com/moses-smt/mgiza.git
cd mgiza/mgizapp
cmake .
In CMakeList.txt line 39, delete -lrt
make
make install
**warning, Moses made for Linux environments and not guaranteed on OSX
https://www.mail-archive.com/moses-support@mit.edu/msg14530.html
**note, I assume Moses is in ~/tools/
**note, Moses looks for dyld images in ~/Desktop/tools/mosesdecoder
**note, Make sure merge_alignment.py is in mgizapp directory
**note, must have -mgiza as a command option when training
"""
import subprocess
import os

from EuroParlParser import EuroParlParser
import utils

NGRAM = 3
NCPUS = 4

class Decoder(object):
    """
    Decoder takes as arguments a portion float which indicatates the
    percentage of data to use, 4 language directories, and an option
    parameter that forces the creation of new data subsets
    """
    def __init__(self, portion, lang1_dir, lang2_1_dir, lang2_2_dir, \
                 lang3_dir, new_subset=False):

        if new_subset:
            self._new_subsets(lang1_dir, lang2_1_dir, \
                              lang2_2_dir, lang3_dir, portion)

        self._subset_data(lang1_dir, lang2_1_dir, lang2_2_dir, \
                          lang3_dir, portion)
        self._lang_models([lang2_1_dir, lang3_dir])
        is_training = self._train_translation_system(lang1_dir, lang2_1_dir, \
                                                     lang2_2_dir, lang3_dir)
        if is_training:
            return

        # TODO check if previous tuning data exists to skip tuning
        is_tuning = self._tune_system(lang1_dir, lang2_1_dir, \
                                      lang2_2_dir, lang3_dir, portion)
        if is_tuning:
            return

        # Evaluate Bleu scores

    def _new_subsets(self, lang1_dir, lang2_1_dir, lang2_2_dir,
                     lang3_dir, portion):
        """
        Creates a managable sized subset of the data for use.
        """
        self._make_new_subset(lang1_dir, lang2_1_dir, portion)
        self._make_new_subset(lang2_2_dir, lang3_dir, portion)

    def _make_new_subset(self, first_lang_dir, second_lang_dir, portion):
        utils.force_print("Making new data subset\n")

        parser = EuroParlParser(first_lang_dir, second_lang_dir)
        first_lang, second_lang = parser.get_random_subset_corpus(portion)
        self._save_subsets(first_lang_dir, first_lang, second_lang_dir, \
                           second_lang)

    def _save_subsets(self, first_lang_dir, first_lang, second_lang_dir, \
                      second_lang):
        for directory, data in [[first_lang_dir, first_lang],
                                [second_lang_dir, second_lang]]:
            new_data = utils.make_filename_from_filepath(directory)
            datafile = "data/" + new_data + ".subset"
            utils.write_data(data, datafile)

    def _subset_data(self, lang1_dir, lang2_1_dir, lang2_2_dir, \
                     lang3_dir, portion):
        """
        This function manages the subset data. Its subroutines perform
        the following:
        1) Creates non-existant subset data
        2) Loads subset data into class variables
        """
        self._subsets_exist(lang1_dir, lang2_1_dir, portion)
        self._subsets_exist(lang2_2_dir, lang3_dir, portion)
        self._load_subsets([lang1_dir, lang2_1_dir, lang2_2_dir, lang3_dir])

    def _subsets_exist(self, dir1, dir2, portion):
        """
        Given directories, checks if the subsets exist. If not,
        create them here
        """
        if not utils.data_exists("data", "subset", dir1, dir2):
            self._make_new_subset(dir1, dir2, portion)

    def _load_subsets(self, dirs):
        """
        Given a list of dirctories dirs, this function attempts to
        load the .subset files and store them as class variables
        """
        new_class_vars = ['lang1_subset', 'lang2_1subset', \
                          'lang2_2subset', 'lang3_subset']
        for var, d in zip(new_class_vars, dirs):
            subset_datafile = utils.make_filename_from_filepath(d)
            subset = utils.load_data("data/" + subset_datafile + \
                                                ".subset")
            setattr(self, var, subset)

        utils.assert_equal_lens(self.lang1_subset, self.lang2_1subset)
        utils.assert_equal_lens(self.lang2_2subset, self.lang3_subset)

    def _lang_models(self, dirs):
        """
        Ensures language models and their binary counter-parts
        exist in the project directory
        """
        self._lang_models_exist(dirs)
        self._bin_lang_models_exist(dirs)

    def _lang_models_exist(self, dirs):
        if utils.data_exists('lm', 'lm', dirs[0], dirs[1]):
            return
        else:
            self._make_language_models(dirs)

    def _make_language_models(self, dirs):
        """
        Building the language model ensures fluid output. Should only
        be built for the target language, passed as parameters in a list
        In the Moses tutortial, the .lm file corresponds to the .arpa
        file
        """
        utils.make_dir("lm")

        for d in dirs:
            subset_datafile = utils.make_filename_from_filepath(d)
            output_datafile = "lm/" + subset_datafile + ".lm"
            subset_datafile = "data/" + subset_datafile + ".subset"

            command = "/Users/urielmandujano/tools/mosesdecoder/bin/lmplz "\
                  "-o {} < ".format(NGRAM) + subset_datafile + " > " + \
                  output_datafile
            subprocess.call(command, shell=True)

    def _bin_lang_models_exist(self, dirs):
        if utils.data_exists('lm', 'blm', dirs[0], dirs[1]):
            return
        else:
            self._binarizelm(dirs)

    def _binarizelm(self, dirs):
        """
        Binarizes the 2 target language model files for faster loading.
        Recommended for larger languages
        """
        for d in dirs:
            subset_datafile = utils.make_filename_from_filepath(d)
            blm_datafile = "lm/" + subset_datafile + ".blm"
            lm_datafile = "lm/" + subset_datafile + ".lm"

            command = "/Users/urielmandujano/tools/mosesdecoder/bin"\
                      "/build_binary " + lm_datafile + " " + blm_datafile
            subprocess.call(command, shell=True)

    def _train_translation_system(self, lang1_dir, lang2_1_dir, \
                                  lang2_2_dir, lang3_dir):
        """
        Uses MGIZA to perform word alignments, extracts phrases, scores
        phrases, creates lex tables and a Moses config file
        If no training is necessary, Decoder will automatically begin
        tuning. If necessary, will launch training and program will exit
        """
        training = False
        if self._no_previous_training(lang1_dir, lang2_1_dir):
            self._training_comment(lang1_dir, lang2_1_dir)
            self._train(lang1_dir, lang2_1_dir)
            training = True
        if self._no_previous_training(lang2_2_dir, lang3_dir):
            self._training_comment(lang2_2_dir, lang3_dir)
            self._train(lang2_2_dir, lang3_dir)
            training = True
        return training

    def _no_previous_training(self, lang1_dir, lang2_dir):
        """
        Checks if there exists a .ini file in the proper directories
        If so, skips the training step. If not, performs the training
        """
        working_dir = utils.directory_name_from_root(lang1_dir)
        utils.make_dir(working_dir)
        if os.path.exists("{}/train/model/moses.ini".format(working_dir)):
            return False
        return True

    def _train(self, first_lang_dir, second_lang_dir):
        """
        Carries out the training with the correct arguments. Creates
        a suitably named working directory, extracts the information
        for file names needed to run moses correctly, runs moses, and
        returns one directory up
        """
        working_dir = utils.directory_name_from_root(first_lang_dir)
        os.chdir(working_dir)

        filename1 = utils.make_filename_from_filepath(first_lang_dir)
        filename2 = utils.make_filename_from_filepath(second_lang_dir)

        fileroot = "../data/" + utils.get_language_root(filename1)
        file1_ext = utils.get_language_extention(filename1)[1:] + ".subset"
        file2_ext = utils.get_language_extention(filename2)[1:] + ".subset"

        lm = "$HOME/projects/nnmt/lm/" + filename2 + ".blm"
        result = utils.get_language_extention(filename1)[1:] + \
                 utils.get_language_extention(filename2) + ".training"

        trainer = "~/tools/mosesdecoder/scripts/training/train-model.perl"
        command = "nohup nice " + trainer + \
        " -root-dir train -corpus {}".format(fileroot) + \
        " -f {} -e {} -alignment".format(file1_ext, file2_ext) + \
        " grow-diag-final-and -reordering msd-bidirectional-fe" + \
        " -lm 0:3:{}:8".format(lm) + \
        " -cores {}".format(NCPUS) + \
        " -mgiza" + \
        " -external-bin-dir ~/tools/mosesdecoder/tools/mgizapp/" + \
        " >& {} &".format(result)
        subprocess.call(command, shell=True)
        os.chdir("..")

    def _training_comment(self, lang1_dir, lang2_dir):
        """
        Crafts a comment to alert the user that training is occurring
        """
        working_dir = utils.directory_name_from_root(lang1_dir)

        filename1 = utils.make_filename_from_filepath(lang1_dir)
        filename2 = utils.make_filename_from_filepath(lang2_dir)
        root1 = utils.get_language_root(filename1)
        root2 = utils.get_language_root(filename2)

        comment = "Beginning training for {} and {}.".format(root1, root2) + \
        "\nWill take a while. Check mgiza's status using ps\n"
        utils.force_print(comment)

    def _tune_system(self, lang1_dir, lang2_1_dir, \
                     lang2_2_dir, lang3_dir, portion):
        """
        Creates a tuning dataset from the total data. Note, this is not
        a true tuning set. True tuning sets should be entirely seperate
        from training data. However, due to lack of computing resources,
        I use a random subset of the total data as an acceptable substitute.
        Then, performs tuning using mosesdecoder
        """
        self._tuning_data_exists(lang1_dir, lang2_1_dir, portion)
        self._tuning_data_exists(lang2_2_dir, lang3_dir, portion)
        self._tune_set(lang1_dir, lang2_1_dir)
        self._tune_set(lang2_2_dir, lang3_dir)

    def _tuning_data_exists(self, first_lang_dir, second_lang_dir, portion):
        if utils.data_exists('data', 'tune', first_lang_dir, second_lang_dir):
            return
        else:
            self._make_new_tune_set(first_lang_dir, second_lang_dir, portion)

    def _make_new_tune_set(self, first_lang_dir, second_lang_dir, portion):
        utils.force_print("Making new tuning subset\n")
        parser = EuroParlParser(first_lang_dir, second_lang_dir)
        first_lang, second_lang = parser.get_random_subset_corpus(portion*.1)
        self._save_tuning_set(first_lang_dir, first_lang, second_lang_dir, \
                              second_lang)

    def _save_tuning_set(self, first_lang_dir, first_lang, second_lang_dir, \
                         second_lang):
        """
        Saves the data for the tuning set as a .tune file extension
        """
        for dire, data in [[first_lang_dir, first_lang],
                           [second_lang_dir, second_lang]]:
            new_data = utils.make_filename_from_filepath(dire)
            datafile = "data/" + new_data + ".tune"
            utils.write_data(data, datafile)

    def _tune_set(self, first_lang_dir, second_lang_dir):
        """
        Initiates the tuning routine. This will take a while. Changes
        working directory into the appropriate directory, executes the
        command and returns to the project's base directory.
        """
        working_dir = utils.directory_name_from_root(first_lang_dir)
        os.chdir(working_dir)

        tune_data1 = utils.make_filename_from_filepath(first_lang_dir)
        tune_data1 = "/Users/urielmandujano/projects/nnmt/data/" + \
                     tune_data1 + ".tune"
        tune_data2 = utils.make_filename_from_filepath(second_lang_dir)
        tune_data2 = "/Users/urielmandujano/projects/nnmt/data/" + \
                     tune_data2 + ".tune"

        command = "nohup nice" + \
        " ~/tools/mosesdecoder/scripts/training/mert-moses.pl" + \
        " {} {}".format(tune_data1, tune_data2) + \
        " ~/tools/mosesdecoder/bin/moses train/model/moses.ini" + \
        " --mertdir ~/tools/mosesdecoder/bin/" + \
        ' --decoder-flags="-threads {}"'.format(NCPUS) + \
        " &> mert.out &"
        subprocess.call(command, shell=True)
        os.chdir("..")

def main():
    lang1 = '/Users/urielmandujano/data/europarl/europarl-v7.es-en.es'
    lang2_1 = '/Users/urielmandujano/data/europarl/europarl-v7.es-en.en'
    lang2_2 = '/Users/urielmandujano/data/europarl/europarl-v7.fr-en.en'
    lang3 = '/Users/urielmandujano/data/europarl/europarl-v7.fr-en.fr'
    decoder = Decoder(.1, lang1, lang2_1, lang2_2, lang3)

if __name__ == "__main__":
    main()
