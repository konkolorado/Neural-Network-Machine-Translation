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

-- Installing CMPH library for efficient table loading
install cmph from http://sourceforge.net/projects/cmph/
cd /path/tp/cmph/
./configure; make; make install

-- Compiling moses
Install
cd /path/to/moses/
./bjam --with-cmph=/Users/urielmandujano/tools/cmph-2.0 --with-boost=/usr/local/boost_1_61_0/ -j4 toolset=clang --with-xmlrpc-c=/usr/local

**note, I assume Moses is in ~/tools/
**note, Moses looks for dyld images in ~/Desktop/tools/mosesdecoder
**note, must have -mgiza as a command option when training

"""
import subprocess
import os
import sys

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

        is_tuning = self._tune_system(lang1_dir, lang2_1_dir, \
                                      lang2_2_dir, lang3_dir, portion)
        if is_tuning:
            return

        self._test(lang1_dir, lang2_1_dir, lang2_2_dir, lang3_dir, portion)

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

        lm = "$PWD/../lm/" + filename2 + ".blm"
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
        "\nWill take a while. Check mgiza's status using ps.\nRerun" + \
        " python Decoder.py when complete.\n"
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
        tuning = False
        if not self._already_tuned(lang1_dir):
            self._tuning_data_exists(lang1_dir, lang2_1_dir, portion)
            self._tune_set(lang1_dir, lang2_1_dir)
            tuning = True

        if not self._already_tuned(lang2_2_dir):
            self._tuning_data_exists(lang2_2_dir, lang3_dir, portion)
            self._tune_set(lang2_2_dir, lang3_dir)
            tuning = True
        return tuning

    def _already_tuned(self, first_lang_dir):
        """
        Checks for a moses.ini file in the proper location and reports
        True if found. This way, you don't have to re-tune the system
        """
        working_dir = utils.directory_name_from_root(first_lang_dir)
        filename = working_dir + "/mert-work/moses.ini"
        return os.path.exists(filename)

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
        self._tuning_comment(first_lang_dir, second_lang_dir)
        working_dir = utils.directory_name_from_root(first_lang_dir)
        os.chdir(working_dir)

        tune_data1 = utils.make_filename_from_filepath(first_lang_dir)
        tune_data1 = "$PWD/../data/" + tune_data1 + ".tune"
        tune_data2 = utils.make_filename_from_filepath(second_lang_dir)
        tune_data2 = "$PWD/../data/" + tune_data2 + ".tune"

        command = "nohup nice" + \
        " ~/tools/mosesdecoder/scripts/training/mert-moses.pl" + \
        " {} {}".format(tune_data1, tune_data2) + \
        " ~/tools/mosesdecoder/bin/moses train/model/moses.ini" + \
        " --mertdir ~/tools/mosesdecoder/bin/" + \
        ' --decoder-flags="-threads {}"'.format(NCPUS) + \
        " &> mert.out &"
        subprocess.call(command, shell=True)
        os.chdir("..")

    def _tuning_comment(self, lang1_dir, lang2_dir):
        """
        Crafts a comment to alert the user that tuning is occurring
        """
        working_dir = utils.directory_name_from_root(lang1_dir)

        filename1 = utils.make_filename_from_filepath(lang1_dir)
        filename2 = utils.make_filename_from_filepath(lang2_dir)
        root1 = utils.get_language_root(filename1)
        root2 = utils.get_language_root(filename2)

        comment = "Beginning tuning for {} and {}.".format(root1, root2) + \
        "\nWill take a while. Check status using ps.\nRerun" + \
        " python Decoder.py when complete.\n"
        utils.force_print(comment)

    def _test(self, lang1_dir, lang2_1_dir, \
              lang2_2_dir, lang3_dir, portion):
        """
        Tests the resultant translation systems on some held out data
        To do so, it first ensures testing data exists
        """
        self._test_data_exists(lang1_dir, lang2_1_dir, \
                               lang2_2_dir, lang3_dir, portion)

        """
        self._binarized_phrase_table_exists(lang1_dir)
        self._binarized_phrase_table_exists(lang2_2_dir)

        self._update_config_file(lang1_dir)
        self._update_config_file(lang2_2_dir)
        """

        # Filter and binarize the phrase + reordering-table for 1st leg
        test_src = "../data/" + \
                   utils.make_filename_from_filepath(lang1_dir) + \
                   ".test"
        self._filter_test_set(lang1_dir, test_src)

        # Translate for first leg
        if self._no_translation(lang1_dir):
            self._perform_pivot_translation(lang1_dir)


        # Filter and binarize the phrase + reordering-table for 2nd leg
        test_tar = "../data/" + \
                   utils.make_filename_from_filepath(lang1_dir) + \
                   ".translated"
        self._filter_test_set(lang2_2_dir, test_tar)

        # Translate for second leg
        if self._no_translation(lang2_2_dir):
            self._perform_pivot_translation(lang2_2_dir)

        # TODO
        # Opt1: Get bleu scores for each leg of pivot
        self._get_bleu_scores(lang2_2_dir, lang3_dir)

    def _test_data_exists(self, first_lang_dir, second_lang_dir, \
                          third_lang_dir, fourth_lang_dir, portion):
        if utils.data_exists('data', 'test', first_lang_dir, \
                             second_lang_dir) and \
           utils.data_exists('data', 'test', third_lang_dir, \
                             fourth_lang_dir):
            return
        else:
            self._make_new_test_set(first_lang_dir, second_lang_dir, \
                                    third_lang_dir, fourth_lang_dir, portion)

    def _make_new_test_set(self, first_lang_dir, second_lang_dir, \
                           third_lang_dir, fourth_lang_dir, portion):
        """
        Makes new test set at 10% of total training data used. Here the
        system takes a random sample, in an ideal scenario this subset
        would be disjoint from the test/tune set. But my low-computational
        power scenario is real
        """
        utils.force_print("Making new testing subset\n")
        self._match(first_lang_dir, second_lang_dir, third_lang_dir, \
                    fourth_lang_dir)



        #self._match(parser1, parser2)

        #first_lang, second_lang = parser.make_test_data([], portion * .1)
        #sample_indices = parser.sample_indices
        #third_lang, fourth_lang = parser2.make_test_data(sample_indices, \
        #                                                portion*.1)
        #print first_lang[:5]
        #print second_lang[:5]
        #print third_lang[:5]
        #print fourth_lang[:5]
        while True:
            continue

        self._save_testing_set(first_lang_dir, first_lang, second_lang_dir, \
                               second_lang)
        self._save_testing_set(third_lang_dir, third_lang, fourth_lang_dir, \
                               fourth_lang)


    def _mangage_matching(self, size, first_lang_dir, second_lang_dir, \
                           third_lang_dir, fourth_lang_dir):
        """
        Performs the matching routine to make sure the same sentences are
        represented in both test sets
        """
        utils.force_print("Matching datasets\n")

        for i in range(1, int(1 / size)+1 ):
            parser1 = EuroParlParser(first_lang_dir, second_lang_dir)
            self._reduce_data(parser1, i, size)
            parser2 = EuroParlParser(third_lang_dir, fourth_lang_dir)
            self._reduce_data(parser2, i, size)
            shared_indices = self._get_matching(parser1, parser2)
            self._save_matches(parser1, parser2, shared_indices)
            del parser1
            del parser2

    def _save_matches(self, p1, p2, shared):
        """
        Writes the matched indices to a filename ending in .matched
        """
        p1_lang1, p1_lang2 = [], []
        p2_lang1, p2_lang2 = [], []
        for (p1_i, p2_i) in shared:
            p1_lang1.append(p1.lang1_cleansed[p1_i])
            p1_lang2.append(p1.lang2_cleansed[p1_i])
            p2_lang1.append(p2.lang1_cleansed[p2_i])
            p2_lang2.append(p2.lang2_cleansed[p2_i])

        for directory, data in [[p1.lang1_dir, p1_lang1],
                                [p1.lang2_dir, p1_lang2],
                                [p2.lang1_dir, p2_lang1],
                                [p2.lang2_dir, p2_lang2]]:
            new_name = utils.make_filename_from_filepath(directory)
            new_name = "data/" + new_name + ".matched"
            if os.path.exists(new_name):
                utils.add_data(data, new_name)
            else:
                utils.write_data(data, new_name)
            del data

    def _get_matching(self, p1, p2):
        """
        Actually modifies the parser data so that the only data
        available is the one shared by both parsers
        """
        p1_lang2_dict = {}
        for i, sentence in enumerate(p1.lang2_cleansed):
            p1_lang2_dict[sentence] = i

        shared = []
        for i, sentence in enumerate(p2.lang1_cleansed):
            if sentence in p1_lang2_dict:
                shared.append((p1_lang2_dict[sentence], i))
        return shared

    def _reduce_data(self, parser, section, percentage):
        """
        This matching problem takes too much memory to do all at once.
        Here, I cut the data down into smaller chunks (percentage).
        The parameter section indicates which section of the data
        to work on i.e. 1st 25%, 2nd 35%... and adjusts the parsers
        data accordingly
        """
        size = len(parser.lang1_cleansed)
        bottom_limit = int(size * percentage * (section - 1))
        uppper_limit = int(bottom_limit + size * percentage)

        lang1_cleansed = parser.lang1_cleansed[bottom_limit:uppper_limit][:]
        del parser.lang1_cleansed
        lang2_cleansed = parser.lang2_cleansed[bottom_limit:uppper_limit][:]
        del parser.lang2_cleansed

        parser.lang1_cleansed = lang1_cleansed
        parser.lang2_cleansed = lang2_cleansed

    def _match(self, p1_l1_dir, p1_l2_dir, p2_l1_dir, p2_l2_dir):
        """
        Takes 2 parser objects as params. Makes sure that the two objects
        share the same cleansed data (Examples seen in es-en are
        present in en-fr in a 1-to-1 relationship)
        """
        if self._matched_files_exist([p1_l1_dir, p1_l2_dir, \
                                      p2_l1_dir, p2_l2_dir]):
            print "loading matched files"
            #self._load_matched_data()
        else:
            self._mangage_matching(.5, p1_l1_dir, p1_l2_dir, \
                                   p2_l1_dir, p2_l2_dir)

    def _matched_files_exist(self, dirs):
        """
        Checks if data files ending in '.matched extension exist'
        """
        return utils.data_exists('data', 'matched', dirs[0], dirs[1]) and \
           utils.data_exists('data', 'matched', dirs[2], dirs[3])

    def _save_testing_set(self, first_lang_dir, first_lang, second_lang_dir, \
                          second_lang):
        """
        Saves newly created testing data in data directory using the
        .test extension
        """
        for dire, data in [[first_lang_dir, first_lang],
                           [second_lang_dir, second_lang]]:
            new_data = utils.make_filename_from_filepath(dire)
            datafile = "data/" + new_data + ".test"
            utils.write_data(data, datafile)

    def _binarized_phrase_table_exists(self, first_lang_dir):
        """
        Binarize the phrase table for faster loading during the testing
        process
        """
        working_dir = utils.directory_name_from_root(first_lang_dir)
        utils.make_dir(working_dir + "/binarized-model/")

        if not os.path.exists(working_dir + \
                              "/binarized-model/phrase-table.minphr"):
            utils.force_print("Making phrase-table\n")
            self._binarize_phrase_table(working_dir)

        if not os.path.exists(working_dir + \
                              "/binarized-model/reordering-table.minlexr"):
            utils.force_print("Making reordering-table\n")
            self._binarize_reordering_table(working_dir)

    def _binarize_phrase_table(self, working_dir):
        os.chdir(working_dir)
        command = "~/tools/mosesdecoder/bin/processPhraseTableMin" + \
        " -in train/model/phrase-table.gz -nscores 4 " + \
        " -out binarized-model/phrase-table -threads {}".format(NCPUS)
        subprocess.call(command, shell=True)
        os.chdir("..")

    def _binarize_reordering_table(self, working_dir):
        os.chdir(working_dir)
        command =  "~/tools/mosesdecoder/bin/processLexicalTableMin " + \
                   " -in train/model/reordering-table.wbe-msd-" + \
                   "bidirectional-fe.gz" + \
                   " -out binarized-model/reordering-table" + \
                   " -threads {}".format(NCPUS)
        subprocess.call(command, shell=True)
        os.chdir("..")

    def _update_config_file(self, lang_dir):
        """
        After creating the binarized phrase table, the old moses.ini
        config file created during tuning must be updated to point
        to the new binarized phrase table. This does that using sed
        """
        working_dir = utils.directory_name_from_root(lang_dir)
        config = working_dir + "/mert-work/moses.ini"

        command1 = "sed -i.bak " + \
                   "s/PhraseDictionaryMemory/PhraseDictionaryCompact/" + \
                   " {}/mert-work/moses.ini".format(working_dir)
        subprocess.call(command1, shell=True)

        command2 = "sed -i.bak " + \
                   "'s/PhraseDictionaryCompact\(.*\){}.*input-factor\(.*\)"\
                                                    .format(working_dir) + \
                   "/PhraseDictionaryCompact\\1{}\/binarized-model\/"\
                                                    .format(working_dir) + \
                "phrase-table.minphr input-factor\\2/' {}/mert-work/moses.ini"\
                                                    .format(working_dir)
        subprocess.call(command2, shell=True)

        command3 = "sed -i.bak " + \
                   "'s/LexicalReordering \(.*\){}.*".format(working_dir) + \
                   "/LexicalReordering \\1{}\/binarized-model\/"\
                                                    .format(working_dir) + \
                   "reordering-table/' {}/mert-work/moses.ini"\
                                                    .format(working_dir)
        subprocess.call(command3, shell=True)

    def _filter_test_set(self, lang_dir, test_dat):
        """
        Speeds up translation by minimizing the phrase tables to be relevant
        to the given test set. Can be tested by running command
        ~/tools/mosesdecoder/bin/moses -f \
        /path/to/project/es-en.working/binarized-filtered-model/moses.ini \
        -i /path/to/project/es-en.working/binarized-filtered-model/input.### \
        -minlexr-memory
        """
        working_dir = utils.directory_name_from_root(lang_dir)
        os.chdir(working_dir)

        filtered_dat = "binarized-filtered-model/"
        command = "/Users/urielmandujano/tools/mosesdecoder/scripts/" + \
                  "training/filter-model-given-input.pl" + \
                  " {} mert-work/moses.ini".format(filtered_dat) + \
                  " {}".format(test_dat) + \
                  " -Binarizer /Users/urielmandujano/tools/mosesdecoder" + \
                  "/bin/processPhraseTableMin"
        subprocess.call(command, shell=True)
        os.chdir("..")

    def _no_translation(self, lang_dir):
        """
        Determines if the translation has already been performed
        given a language directory
        """
        trans = "data/" + utils.make_filename_from_filepath(lang_dir) + \
                ".translated"
        return not utils.file_exists(trans)

    def _perform_pivot_translation(self, lang_dir):
        """
        Translates from the origin language to pivot language. Outputs
        to a file
        """
        working_dir = utils.directory_name_from_root(lang_dir)
        result = "../data/" + \
                 utils.make_filename_from_filepath(lang_dir) + \
                 ".translated"
        debug = utils.make_filename_from_filepath(lang_dir) + \
                ".translated"
        os.chdir(working_dir)

        in_name = utils.get_full_filename("binarized-filtered-model", "input")
        com ="nohup nice /Users/urielmandujano/tools/mosesdecoder/bin/moses"+\
        " -f binarized-filtered-model/moses.ini <" + \
        " binarized-filtered-model/{}".format(in_name) + \
        " > {}".format(result) + \
        " 2> binarized-filtered-model/{}.out".format(debug) + \
        " -minlexr-memory"
        subprocess.call(com, shell=True)
        os.chdir("..")

    def _get_bleu_scores(self, first_lang_dir, second_lang_dir):
        """
        Runs the moses script to eval bleu scores on a completed translation
        """
        true = "data/" + utils.make_filename_from_filepath(second_lang_dir) + \
               ".test"
        translated = "data/" + \
                     utils.make_filename_from_filepath(first_lang_dir) + \
                     ".translated"
        command = "/Users/urielmandujano/tools/mosesdecoder/scripts/" + \
                  "generic/multi-bleu.perl -lc {}".format(true) + \
                  " < {}".format(translated)
        subprocess.call(command, shell=True)

def main():
    lang1 = '/Users/urielmandujano/data/europarl/europarl-v7.es-en.es'
    lang2_1 = '/Users/urielmandujano/data/europarl/europarl-v7.es-en.en'
    lang2_2 = '/Users/urielmandujano/data/europarl/europarl-v7.fr-en.en'
    lang3 = '/Users/urielmandujano/data/europarl/europarl-v7.fr-en.fr'
    decoder = Decoder(.1, lang1, lang2_1, lang2_2, lang3)

if __name__ == "__main__":
    main()
