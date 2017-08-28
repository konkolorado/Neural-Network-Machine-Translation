"""
Pipelines the creation and evalution of the pivoting SMT process.
"""
import sys
sys.path.append('src')

from Parser import Parser
from Train import Train
from Tune import Tune
from Test import Test

from FileDataPair import FileDataPair

import utilities

def smtp_pipeline(config):
    path_to_moses = config.get("Environment Settings", "path_to_moses_decoder")
    mem_limit = config.getint("Environment Settings", "mem_limit")
    max_len = config.getint("Iteration Settings", "max_sentence_len")
    min_len = config.getint("Iteration Settings", "min_sentence_len")

    srcf = utilities.safe_string(config.get("Iteration Settings", "src_lang_data"))
    piv1f = utilities.safe_string(config.get("Iteration Settings", "src_piv_lang_data"))
    piv2f = utilities.safe_string(config.get("Iteration Settings", "piv_tar_lang_data"))
    tarf = utilities.safe_string(config.get("Iteration Settings", "tar_lang_data"))
    train = config.getfloat("Iteration Settings", "train_split")
    test = config.getfloat("Iteration Settings", "test_split")
    ncpus = config.getint("Environment Settings", "ncpus")
    ngram = config.getint("Environment Settings", "ngram")
    work_dir1 = utilities.safe_string(config.get("Iteration Settings", "working_dir_first_leg"))
    work_dir2 = utilities.safe_string(config.get("Iteration Settings", "working_dir_second_leg"))

    pair1, pair2 = FileDataPair(srcf, piv1f), FileDataPair(piv2f, tarf)
    raw_files = pair1.get_raw_filenames() + pair2.get_raw_filenames()
    pair1_tokenized_src, pair1_tokenized_tar = pair1.get_tokenized_filenames()
    pair2_tokenized_src, pair2_tokenized_tar = pair2.get_tokenized_filenames()
    pair1_cleansed_src, pair1_cleansed_tar = pair1.get_cleansed_filenames()
    pair2_cleansed_src, pair2_cleansed_tar = pair2.get_cleansed_filenames()

    parser = Parser(path_to_moses, mem_limit, max_len, min_len, False)
    parser.tokenize_files(raw_files)
    parser.cleanse(pair1_tokenized_src, pair1_tokenized_tar)
    parser.cleanse(pair2_tokenized_src, pair2_tokenized_tar)
    parser.split_train_tune_test(pair1_cleansed_src, pair1_cleansed_tar, \
        pair2_cleansed_src, pair2_cleansed_tar, train, test)
    parser.match(pair1_test_src, pair2_test_tar, pair2_test_src, pair2_test_tar)

    pair1_target_train_filename = pair1.get_target_train_filename()
    pair2_target_train_filename = pair2.get_target_train_filename()
    pair1_train_src, pair1_train_tar = pair1.get_train_filenames()
    pair2_train_src, pair2_train_tar = pair2.get_train_filenames()

    trainer = Train(path_to_moses, ncpus, ngram, False)
    trainer.build_language_models(pair1_target_train_filename)
    trainer.build_language_models(pair2_target_train_filename)
    trainer.train(pair1_train_src, pair1_train_tar, work_dir1)
    trainer.train(pair2_train_src, pair2_train_tar, work_dir2)

    pair1_tune_src, pair1_tune_tar = pair1.get_tune_filenames()
    pair2_tune_src, pair2_tune_tar = pair2.get_tune_filenames()

    tuner = Tune(path_to_moses, ncpus, False)
    tuner.tune(pair1_tune_src, pair1_tune_tar, work_dir1)
    tuner.tune(pair2_tune_src, pair2_tune_tar, work_dir2)

    pair1_test_src, pair1_test_tar = pair1.get_test_filenames()
    pair2_test_src, pair2_test_tar = pair2.get_test_filenames()
    pair1_test_tar = pair1.get_eval_filename()
    pair2_test_tar = pair2.get_eval_filename()

    test = Test(path_to_moses, False)
    test.test_pivoting_quality(pair1_test_tar, work_dir1,
        pair2_test_tar, work_dir2)

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
