import utilities

class FileData(object):
    def __init__(self, file_uri, home="data/"):
        assert utilities.file_exists(file_uri), f"{file_uri} not found"

        self.raw_file = file_uri
        self.dir = home
        self.train_dir = "train/"
        self.tune_dir = "tune/"
        self.test_dir = "test/"

        self._init_filename()
        self._init_filename_tokenized()
        self._init_filename_cleansed()

        self._init_filename_for_train_set()
        self._init_filename_for_tune_set()
        self._init_filename_for_test_set()
        self._init_filename_for_pivot_evaluation()

    def _init_filename(self):
        self.base_name = self.dir + utilities.strip_filename_from_path(self.raw_file)

    def get_filename(self):
        return self.base_name

    def _init_filename_tokenized(self):
        self.name_tok = self.base_name + ".tok"

    def get_filename_tokenized(self):
        return self.name_tok

    def _init_filename_cleansed(self):
        self.name_cleansed = self.name_tok + ".cleansed"

    def get_filenames_cleansed(self):
        return self.name_cleansed

    def _init_filename_for_train_set(self):
        self.train = self._make_set_filename(self.name_cleansed, self.train_dir)

    def get_filename_train(self):
        return self.train

    def _init_filename_for_tune_set(self):
        self.tune = self._make_set_filename(self.name_cleansed, self.tune_dir)

    def get_filename_tune(self):
        return self.tune

    def _init_filename_for_test_set(self):
        self.test = self._make_set_filename(self.name_cleansed, self.test_dir)

    def get_filename_test(self):
        return self.test

    def _make_set_filename(self, filename, _set):
        ext = "." + self._chop_last_char(_set)
        return self._insert_path_into_name_after_symbol(filename, "/", _set) + ext

    def _insert_path_into_name_after_symbol(self, name, symbol, path):
        index = name.find(symbol) + 1
        return name[:index] + path + name[index:]

    def _chop_last_char(self, string):
        str_limit = len(string) - 1
        return string[:str_limit]

    def _init_filename_for_pivot_evaluation(self):
        self.eval = self.test + ".matched"

    def get_filename_for_pivot_evaluation(self):
        return self.eval

def main():
    fd = FileData("data/src/europarl-v7.es-en.es")
    print(fd.get_filename())
    print(fd.get_filename_tokenized())
    print(fd.get_filenames_cleansed())
    print(fd.get_filename_train())
    print(fd.get_filename_tune())
    print(fd.get_filename_test())

    fd = FileData("data/src/europarl-v7.es-en.en")
    fd = FileData("data/src/europarl-v7.fr-en.en")
    fd = FileData("data/src/europarl-v7.fr-en.fr")


if __name__ == '__main__':
    main()
