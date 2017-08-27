import utilities

class FileNames(object):
    def __init__(self, src, piv1, piv2, tar, destdir):
        self.src_path = src
        self.piv1_path = piv1
        self.piv2_path = piv2
        self.tar_path = tar
        self.dir = destdir

        self._init_filenames()
        self._init_filenames_tokenized()
        self._init_filenames_cleansed()

        self._init_filenames_for_train_set()
        self._init_filenames_for_tune_set()
        self._init_filenames_for_test_set()
        self._init_filenames_for_evaluation()

    def get_filenames_with_path(self):
        return self.src_path, self.piv1_path, self.piv2_path, self.tar_path

    def _init_filenames(self):
        self.src_fname = self.dir + utilities.strip_filename_from_path(self.src_path)
        self.piv1_fname = self.dir + utilities.strip_filename_from_path(self.piv1_path)
        self.piv2_fname = self.dir + utilities.strip_filename_from_path(self.piv2_path)
        self.tar_fname = self.dir + utilities.strip_filename_from_path(self.tar_path)

    def get_filenames(self):
        return self.src_fname, self.piv1_fname, self.piv2_fname, self.tar_fname

    def _init_filenames_tokenized(self):
        self.src_tok = self.src_fname + ".tok"
        self.piv1_tok = self.piv1_fname + ".tok"
        self.piv2_tok = self.piv2_fname + ".tok"
        self.tar_tok = self.tar_fname + ".tok"

    def get_filenames_tokenized(self):
        return self.src_tok, self.piv1_tok, self.piv2_tok, self.tar_tok

    def _init_filenames_cleansed(self):
        self.src_cleansed = self.src_tok + ".cleansed"
        self.piv1_cleansed = self.piv1_tok + ".cleansed"
        self.piv2_cleansed = self.piv2_tok + ".cleansed"
        self.tar_cleansed = self.tar_tok + ".cleansed"

    def get_filenames_cleansed(self):
        return self.src_cleansed, self.piv1_cleansed, self.piv2_cleansed, \
            self.tar_cleansed

    def _init_filenames_for_test_set(self):
        self.src_test = self._create_test_file_name(self.src_cleansed)
        self.piv1_test = self._create_test_file_name(self.piv1_cleansed)
        self.piv2_test = self._create_test_file_name(self.piv2_cleansed)
        self.tar_test = self._create_test_file_name(self.tar_cleansed)

    def _create_test_file_name(self, fname):
        return self._insert_path_into_name_after_symbol(fname, "/", "test/") + \
            ".test"

    def _insert_path_into_name_after_symbol(self, name, symbol, path):
        index = name.find(symbol) + 1
        return name[:index] + path + name[index:]

    def get_filenames_for_test_set(self):
        return self.src_test, self.piv1_test, self.piv2_test, self.tar_test

    def _init_filenames_for_train_set(self):
        self.src_train = self._create_train_file_name(self.src_cleansed)
        self.piv1_train = self._create_train_file_name(self.piv1_cleansed)
        self.piv2_train = self._create_train_file_name(self.piv2_cleansed)
        self.tar_train = self._create_train_file_name(self.tar_cleansed)

    def _create_train_file_name(self, fname):
        return self._insert_path_into_name_after_symbol(fname, "/", "train/") + \
            ".train"

    def get_filenames_for_train_set(self):
        return self.src_train, self.piv1_train, self.piv2_train, self.tar_train

    def _init_filenames_for_tune_set(self):
        self.src_tune = self._create_tune_file_name(self.src_cleansed)
        self.piv1_tune = self._create_tune_file_name(self.piv1_cleansed)
        self.piv2_tune = self._create_tune_file_name(self.piv2_cleansed)
        self.tar_tune = self._create_tune_file_name(self.tar_cleansed)

    def _create_tune_file_name(self, fname):
        return self._insert_path_into_name_after_symbol(fname, "/", "tune/") + \
            ".tune"

    def get_filenames_for_tune_set(self):
        return self.src_tune, self.piv1_tune, self.piv2_tune, self.tar_tune

    def _init_filenames_for_evaluation(self):
        self.src_eval = self._create_eval_file_name(self.src_test)
        self.tar_eval = self._create_eval_file_name(self.tar_test)

    def _create_eval_file_name(self, fname):
        return fname + ".matched"

    def get_filenames_for_pivot_evaluation(self):
        return self.src_eval, self.tar_eval

def main():
    fnames = FileNames("data/src/europarl-v7.es-en.es",
        "data/src/europarl-v7.es-en.en", "data/src/europarl-v7.fr-en.en",
        "data/src/europarl-v7.fr-en.fr", "data/")

    print(fnames.get_filenames_with_path())
    print(fnames.get_filenames())
    print(fnames.get_filenames_tokenized())


if __name__ == '__main__':
    main()
