from FileData import *

class FileDataPair(object):
    def __init__(self, src_uri, tar_uri, home="data/"):
        assert utilities.same_until_char(src_uri, tar_uri, "."), "Invalid file names"
        self.pair = FileData(src_uri, home), FileData(tar_uri, home)

    def get_pair(self):
        return self.pair

    def get_source(self):
        return self.pair[0]

    def get_target(self):
        return self.pair[1]

    def get_raw_filenames(self):
        return self.pair[0].raw_file, self.pair[1].raw_file

    def get_tokenized_filenames(self):
        return self.pair[0].get_filename_tokenized(), \
            self.pair[1].get_filename_tokenized()

    def get_cleansed_filenames(self):
        return self.pair[0].get_filenames_cleansed(), \
            self.pair[1].get_filenames_cleansed()

    def get_train_filenames(self):
        return self.pair[0].get_filename_train(), \
            self.pair[1].get_filename_train()

    def get_target_train_filename(self):
        return self.pair[1].get_filename_train()

    def get_tune_filenames(self):
        return self.pair[0].get_filename_tune(), \
            self.pair[1].get_filename_tune()

    def get_test_filenames(self):
        return self.pair[0].get_filename_test(), \
            self.pair[1].get_filename_test()

    def get_eval_filename(self):
        return self.pair[1].get_filename_for_pivot_evaluation()
