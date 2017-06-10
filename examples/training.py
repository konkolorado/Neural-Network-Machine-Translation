#python3.6

"""
Contains example code for using the trainer in Train.py
"""

from Train import Train

def main():

    # Create Train instance, set verbose to True to see whats happening
    trainer = Train(True)

    # Build target language models for only the target languages. In this
    # scenario, the desired target languages are the pivot language in the
    # source to pivot leg of the translation and the target language in the
    # pivot to target leg of the scenario. Language models are saved in the
    # lm directory
    trainer.build_language_models("data/train/europarl-v7.es-en.en.tok.cleansed.train")
    trainer.build_language_models("data/train/europarl-v7.fr-en.fr.tok.cleansed.train")

    # Train each leg of the translation system seperately. The first
    # parameter must be the path to the source training data, the second
    # will be the path to the pivot training data, and the third is the
    # name for the directory which will store the system's results.
    trainer.train("data/train/europarl-v7.es-en.es.tok.cleansed.train",
        "data/train/europarl-v7.es-en.en.tok.cleansed.train", "es-en.working")
    trainer.train("data/train/europarl-v7.fr-en.en.tok.cleansed.train",
        "data/train/europarl-v7.fr-en.fr.tok.cleansed.train", "en-fr.working")


if __name__ == '__main__':
    main()
