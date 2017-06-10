# python3.6

"""
Pipelines the creation and evalution of the pivoting SMT process.
"""

from Parser import Parser
from Train import Train
from Tune import Tune
from Test import Test

import utilities

def smtp_pipeline(srcf, piv1f, work_dir1, piv2f, tarf, work_dir2, train, test):
    parser = Parser(False)
    parser.tokenize(srcf)
    parser.tokenize(piv1f)
    parser.tokenize(piv2f)
    parser.tokenize(tarf)

    srcf = "data/" + utilities.strip_filename_from_path(srcf) + ".tok"
    piv1f = "data/" + utilities.strip_filename_from_path(piv1f) + ".tok"
    piv2f = "data/" + utilities.strip_filename_from_path(piv2f) + ".tok"
    tarf = "data/" + utilities.strip_filename_from_path(tarf) + ".tok"
    parser.cleanse(srcf, piv1f)
    parser.cleanse(piv2f, tarf)

    srcf = srcf + ".cleansed"
    piv1f = piv1f + ".cleansed"
    piv2f = piv2f + ".cleansed"
    tarf = tarf + ".cleansed"
    parser.split_train_tune_test(srcf, piv1f, piv2f, tarf, train, test)

    srcf_test = "data/test/" + utilities.strip_filename_from_path(srcf) + ".test"
    piv1f_test = "data/test/" + utilities.strip_filename_from_path(piv1f) + ".test"
    piv2f_test = "data/test/" + utilities.strip_filename_from_path(piv2f) + ".test"
    tarf_test = "data/test/" + utilities.strip_filename_from_path(tarf) + ".test"
    parser.match(srcf_test, piv1f_test, piv2f_test, tarf_test)

    trainer = Train(False)
    srcf_train = "data/train/" + utilities.strip_filename_from_path(srcf) + ".train"
    piv1f_train = "data/train/" + utilities.strip_filename_from_path(piv1f) + ".train"
    piv2f_train = "data/train/" + utilities.strip_filename_from_path(piv2f) + ".train"
    tarf_train = "data/train/" + utilities.strip_filename_from_path(tarf) + ".train"

    trainer.build_language_models(piv1f_train)
    trainer.build_language_models(tarf_train)
    trainer.train(srcf_train, piv1f_train, work_dir1)
    trainer.train(piv2f_train, tarf_train, work_dir2)

    tuner = Tune(False)
    srcf_tune = "data/tune/" + utilities.strip_filename_from_path(srcf) + ".tune"
    piv1f_tune = "data/tune/" + utilities.strip_filename_from_path(piv1f) + ".tune"
    piv2f_tune = "data/tune/" + utilities.strip_filename_from_path(piv2f) + ".tune"
    tarf_tune = "data/tune/" + utilities.strip_filename_from_path(tarf) + ".tune"

    tuner.tune(srcf_tune, piv1f_tune, work_dir1)
    tuner.tune(piv2f_tune, tarf_train, work_dir2)

    test = Test(False)
    test.test_pivoting_quality(srcf_test + ".matched",
        work_dir1, tarf_test + ".matched", work_dir2)

def main():
    # Supply your own source file, pivot one file, pivot two file, and target
    # file and the directory names in which to save the training in config.ini.
    # Modify train and test percentage (1 - train - test = tune percentage)
    # The filenames need to have same beginnings, up until a ".", after which
    # they may be different

    config = utilities.config_file_reader()
    srcf = utilities.safe_string(config.get("Iteration Settings", "src_lang_data"))
    piv1f = utilities.safe_string(config.get("Iteration Settings", "src_piv_lang_data"))
    piv2f = utilities.safe_string(config.get("Iteration Settings", "piv_tar_lang_data"))
    tarf = utilities.safe_string(config.get("Iteration Settings", "tar_lang_data"))
    work_dir1 = utilities.safe_string(config.get("Iteration Settings", "working_dir_first_leg"))
    work_dir2 = utilities.safe_string(config.get("Iteration Settings", "working_dir_second_leg"))
    train = config.getfloat("Iteration Settings", "train_split")
    test = config.getfloat("Iteration Settings", "test_split")
    smtp_pipeline(srcf, piv1f, work_dir1, piv2f, tarf, work_dir2, train, test)

if __name__ == '__main__':
    main()
