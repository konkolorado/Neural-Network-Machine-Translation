#python3.6

"""
Contains example code for using the tuner in Tune.py
"""

from Tune import Tune

def main():
    # Create Tune instance, set verbose to True to see whats happening
    tuner = Tune(True)

    # Tune each leg of the translation system seperately. The first
    # parameter must be the path to the source tuning data, the second
    # will be the path to the pivot tuning data, and the third is the
    # name of the directory containing the trained model (should be
    # the same as the directory option passed into train.train).
    # Note that the tuning stage is the most time consuming phase
    # of the operation.
    tuner.tune("data/tune/europarl-v7.es-en.es.tok.cleansed.tune",
        "data/tune/europarl-v7.es-en.en.tok.cleansed.tune", "es-en.working")
    tuner.tune("data/tune/europarl-v7.fr-en.en.tok.cleansed.tune",
        "data/tune/europarl-v7.fr-en.fr.tok.cleansed.tune", "en-fr.working")

if __name__ == '__main__':
    main()
