#python3.6

"""
Contains example code for using the test functionality in Test.py
"""

from Test import Test

def interactive_mode():
    # Create Test instance, set verbose to True to see whats happening
    test = Test(True)

    # After training and tuning each leg of the translation pivot, you now have
    # three fully SMT systems: Source to pivot, pivot to target, and source to
    # target. Use the following methods to setup a running Moses decoder server
    # and interactively translate phrases and even sentences. To run each
    # pivot leg interactively, supply the working directory that contains
    # the trained and tuned models (should be the same as the directory passed
    # into the train and tune methods). To run the full translation via
    # pivoting, supply both directories for the source to pivot model and
    # the pivot to test model (in that order)
    test.test_translator_interactive("es-en.working")
    test.test_translator_interactive("en-fr.working")
    test.test_pivoting_interactive("es-en.working", "en-fr.working")

def evaluate_translations():
    # Create Test instance, set verbose to True to see whats happening
    test = Test(True)

    # We can evaluate the tested and tuned model for a measure of correctness
    # by calling the test_translation_quality method and supplying the source
    # language test data, the target languages (true) test data and the
    # directory containing the model. This procedure will output a Bleu score,
    # a score which essentially compares n-grams from the translation to the
    # true output.  The higher the Bleu, the "better" the translation.
    test.test_translation_quality("data/test/europarl-v7.es-en.es.tok.cleansed.test",
        "data/test/europarl-v7.es-en.en.tok.cleansed.test", "es-en.working")
    test.test_translation_quality("data/test/europarl-v7.fr-en.en.tok.cleansed.test",
        "data/test/europarl-v7.fr-en.fr.tok.cleansed.test", "en-fr.working")

def evaluate_pivoting():
    # Create Test instance, set verbose to True to see whats happening
    test = Test(True)

    # We can finally evalute the translations the pivoting provides by
    # calling test_pivoting_quality and providing the source language test file,
    # the directory containing the trained translation model for the source to
    # pivot languages, the target language (true) test file and the directory
    # containing the trained translation model for the pivot to target language.
    # To obtain an accurat scoring, the data files provided MUST be matched.
    # The output is a Bleu score regarding translation quality. Generally,
    # higher scoring is better.
    test.test_pivoting_quality("data/test/europarl-v7.es-en.es.tok.cleansed.test.matched",
        "es-en.working", "data/test/europarl-v7.fr-en.fr.tok.cleansed.test.matched", "en-fr.working")

def main():
    interactive_mode()
    evaluate_translations()
    evaluate_pivoting()


if __name__ == '__main__':
    main()
