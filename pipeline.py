"""
Pipelines the creation and evalution of the pivoting SMT process.
"""
import sys
sys.path.append('src')

from Parser import Parser
from Train import Train
from Tune import Tune
from Test import Test
from FileNames import FileNames

import utilities

def smtp_pipeline(config):
    mem_limit = config.getint("Environment Settings", "mem_limit")
    max_len = config.getint("Iteration Settings", "max_sentence_len")
    min_len = config.getint("Iteration Settings", "min_sentence_len")

    srcf = utilities.safe_string(config.get("Iteration Settings", "src_lang_data"))
    piv1f = utilities.safe_string(config.get("Iteration Settings", "src_piv_lang_data"))
    piv2f = utilities.safe_string(config.get("Iteration Settings", "piv_tar_lang_data"))
    tarf = utilities.safe_string(config.get("Iteration Settings", "tar_lang_data"))
    fnames = FileNames(srcf, piv1f, piv2f, tarf, "data/")

    parser = Parser(mem_limit, max_len, min_len, False)
    parser.tokenize_files(fnames.get_filenames())

    lang_pair1, lang_pair2 = slice(0,2), slice(2,4)
    parser.cleanse(*fnames.get_filenames_tokenized()[lang_pair1])
    parser.cleanse(*fnames.get_filenames_tokenized()[lang_pair2])

    train = config.getfloat("Iteration Settings", "train_split")
    test = config.getfloat("Iteration Settings", "test_split")
    parser.split_train_tune_test(*fnames.get_filenames_cleansed(), train, test)

    test_set_names = fnames.get_filenames_for_test_set()
    parser.match(*test_set_names)

    ncpus = config.getint("Environment Settings", "ncpus")
    ngram = config.getint("Environment Settings", "ngram")
    trainer = Train(ncpus, ngram, False)

    train_set_names = fnames.get_filenames_for_train_set()
    lang_model_tar1, lang_model_tar2 = slice(1,2), slice(3,4)
    lang_pair_train1, lang_pair_train2 = slice(0, 2), slice(2,4)
    trainer.build_language_models(*train_set_names[lang_model_tar1])
    trainer.build_language_models(*train_set_names[lang_model_tar2])

    work_dir1 = utilities.safe_string(config.get("Iteration Settings", "working_dir_first_leg"))
    work_dir2 = utilities.safe_string(config.get("Iteration Settings", "working_dir_second_leg"))
    trainer.train(*train_set_names[lang_pair_train1], work_dir1)
    trainer.train(*train_set_names[lang_pair_train2], work_dir2)

    tuner = Tune(ncpus, False)
    tune_set_names = fnames.get_filenames_for_tune_set()
    lang_pair_tune1, lang_pair_tune2 = slice(0, 2), slice(2, 4)
    tuner.tune(*tune_set_names[lang_pair_tune1], work_dir1)
    tuner.tune(*tune_set_names[lang_pair_tune2], work_dir2)

    test = Test(False)
    evalution_files = fnames.get_filenames_for_pivot_evaluation()
    src_eval_file, tar_eval_file = slice(0, 1), slice(1, 2)
    test.test_pivoting_quality(*evalution_files[src_eval_file], work_dir1,
        *evalution_files[tar_eval_file], work_dir2)

def main():
    # Supply your own source file, pivot one file, pivot two file, and target
    # file and the directory names in which to save the training in config.ini.
    # Modify train and test percentage (1 - train - test = tune percentage)
    # The filenames need to have same beginnings, up until a ".", after which
    # they may be different
    config = utilities.config_file_reader()
    smtp_pipeline(config)

if __name__ == '__main__':
    main()
