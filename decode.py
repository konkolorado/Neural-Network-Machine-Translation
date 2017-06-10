#python3.6

from Parser import Parser
from Train import Train
from Tune import Tune
from Test import Test

"""
USAGE NOTES
- Files need to have same beginnings, up until a ".", after which they may be different
  This isn't my fault, this is just how moses needs its files to be set up
    ex. file1.en file1.es -- ok
        file1.en file2.es -- not ok
        file1.en file1.agrarerger -- ok
"""

def main():
    parser = Parser(True)

    # Tokenize the data
    parser.tokenize("src/europarl-v7.es-en.es")
    parser.tokenize("src/europarl-v7.es-en.en")
    parser.tokenize("src/europarl-v7.fr-en.en")
    parser.tokenize("src/europarl-v7.fr-en.fr")

    # Normalize the data
    parser.cleanse("data/europarl-v7.es-en.es.tok", "data/europarl-v7.es-en.en.tok")
    parser.cleanse("data/europarl-v7.fr-en.en.tok", "data/europarl-v7.fr-en.fr.tok")

    # Split data into train, tune, test sets
    parser.split_train_tune_test("data/europarl-v7.es-en.es.tok.cleansed", "data/europarl-v7.es-en.en.tok.cleansed",
        "data/europarl-v7.fr-en.en.tok.cleansed", "data/europarl-v7.fr-en.fr.tok.cleansed", .6, .2)

    parser.match("data/test/europarl-v7.es-en.es.tok.cleansed.test", "data/test/europarl-v7.es-en.en.tok.cleansed.test",
        "data/test/europarl-v7.fr-en.en.tok.cleansed.test", "data/test/europarl-v7.fr-en.fr.tok.cleansed.test")

    trainer = Train(True)
    # Build target language models
    trainer.build_language_models("data/train/europarl-v7.es-en.en.tok.cleansed.train")
    trainer.build_language_models("data/train/europarl-v7.fr-en.fr.tok.cleansed.train")

    # Train each leg of the translation system
    trainer.train("data/train/europarl-v7.es-en.es.tok.cleansed.train",
        "data/train/europarl-v7.es-en.en.tok.cleansed.train", "es-en.working")
    trainer.train("data/train/europarl-v7.fr-en.en.tok.cleansed.train",
        "data/train/europarl-v7.fr-en.fr.tok.cleansed.train", "en-fr.working")

    # Tune the system on held out data
    tuner = Tune(True)
    tuner.tune("data/tune/europarl-v7.es-en.es.tok.cleansed.tune",
        "data/tune/europarl-v7.es-en.en.tok.cleansed.tune", "es-en.working")
    tuner.tune("data/tune/europarl-v7.fr-en.en.tok.cleansed.tune",
        "data/tune/europarl-v7.fr-en.fr.tok.cleansed.tune", "en-fr.working")

    test = Test(True)
    # Run interactive translator server
    test.test_translator_interactive("es-en.working")
    test.test_translator_interactive("en-fr.working")

    # Score translation quality between pivot translations using held out test data
    test.test_translation_quality("data/test/europarl-v7.es-en.es.tok.cleansed.test",
        "data/test/europarl-v7.es-en.en.tok.cleansed.test", "es-en.working")
    test.test_translation_quality("data/test/europarl-v7.fr-en.en.tok.cleansed.test",
        "data/test/europarl-v7.fr-en.fr.tok.cleansed.test", "en-fr.working")
    # Run interactive translator on pivoting system
    test.test_pivoting_interactive("es-en.working", "en-fr.working")

    # Score translation quality on entire translation using matched test data
    test.test_pivoting_quality("data/test/europarl-v7.es-en.es.tok.cleansed.test.matched",
        "es-en.working", "data/test/europarl-v7.fr-en.fr.tok.cleansed.test.matched", "en-fr.working")

if __name__ == '__main__':
    main()
