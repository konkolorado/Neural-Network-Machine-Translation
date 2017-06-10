#python3.6

"""
Contains example code for using the Parser in Parser.py
"""

from Parser import Parser

def main():
    # Create Parser instance, set verbose to True to see whats happening
    parser = Parser(True)

    # Call tokenize on data path/files. The parser defaults to
    # saving the data in the 'data/' subdirectory.  Tokenized
    # data retains the same filename with an additional .tok
    # file extension added on
    parser.tokenize("src/europarl-v7.es-en.es")
    parser.tokenize("src/europarl-v7.es-en.en")
    parser.tokenize("src/europarl-v7.fr-en.en")
    parser.tokenize("src/europarl-v7.fr-en.fr")

    # Normalizes the data for use with Moses by removing empty lines
    # short lines and long lines.  You must pass two file names
    # (the parallel data, in order of source lang, target lang) so that
    # an inconsistent line in one file is removed in the other as well.
    # These files are saved to the data directory with the .cleansed
    # file extension.
    parser.cleanse("data/europarl-v7.es-en.es.tok", "data/europarl-v7.es-en.en.tok")
    parser.cleanse("data/europarl-v7.fr-en.en.tok", "data/europarl-v7.fr-en.fr.tok")

    # Splits data into train, tune, test sets. Requires 3 file names (both
    # sets of parallel data, in order of source lang, pivot lang, pivot lang
    # target lang) and two decimals, indicating the percentage of the full
    # data to use as training, and testing.  The remainder data will be
    # used as development. All four files must be passed because this
    # method ensures that line x in one file belongs to the same set as
    # line x in the other files. These files are saved in the data/
    # directory under train/ tune/ and test/ sub directories with the
    # appropriate file extension .train, .tune, .test
    parser.split_train_tune_test("data/europarl-v7.es-en.es.tok.cleansed", "data/europarl-v7.es-en.en.tok.cleansed",
        "data/europarl-v7.fr-en.en.tok.cleansed", "data/europarl-v7.fr-en.fr.tok.cleansed", .6, .2)

    # Performs a matching of the data. This step only needs to be completed
    # on the test set. What matching does is it looks at the two pivot
    # language files and keeps only the lines that occur in both files.  Then
    # new source and target language files are created to contain the
    # corresponding lines in the pivot files. This is necessary because when
    # decoding from source to pivot and pivot to target, we only want to
    # translate lines from source to pivot if the pivot to target files contain
    # the "true" translation which we use to score translation quality.
    # These files are saved in the same directory as the file provided with the
    # file extension .matched
    parser.match("data/test/europarl-v7.es-en.es.tok.cleansed.test", "data/test/europarl-v7.es-en.en.tok.cleansed.test",
        "data/test/europarl-v7.fr-en.en.tok.cleansed.test", "data/test/europarl-v7.fr-en.fr.tok.cleansed.test")

if __name__ == '__main__':
    main()
