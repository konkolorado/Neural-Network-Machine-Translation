# Neural-Network-Machine-Translation
MT using Neural Networks between low-data languages. Continuation of work done in JSALT '15.

Unsupervised translation between language pairs only works if there is a lot of data (documents saying the same thing, in 2 different languages). However, not all language pairs have a lot of data to perform high quality translations. The idea in my project is to use a popular language to link language pairs without a lot of data. For example, perhaps Spanish and French don't have much data, but there is a lot of Spanish- English and English-French data. This project attempts to pivot a translation through the 2 language pairs (Spanish->English->French). The goals in this project are as follows:

1) Develop baseline bleu scores for the pivot translation using standard MT tools
Baseline for translating Spanish->English->French:
BLEU = 28.67, 58.7/34.5/22.8/15.4 (BP=0.988, ratio=0.989, hyp_len=526110, ref_len=532206)


2) Develop a baseline bleu score for performing pivoting using Neural Networks

##Getting Started
To begin, you'll need to install moses decoder, a freely available MT system, from

https://github.com/moses-smt/mosesdecoder/tree/RELEASE-2.1.1

Simply download the zip file, unzip it wherever you prefer your moses installation to be. Then, you'll need to install the cmph library to binarize tables later from

http://sourceforge.net/projects/cmph/

and compile by

cd /path/to/cmph/  
./configure; make; make install

Next you'll need to install Giza++ to perform word alignments. You need the boost libraries to compile so let's get those first. Go to

http://www.boost.org/

install boost from website to /usr/local/

Now we can install Giza++ (mgiza)

cd /path/to/mosesdecoder/  
git clone   https://github.com/moses-smt/mgiza.git  
cd mgiza/mgizapp  
cmake .  
In CMakeList.txt line 39, delete -lrt  
make  
make install

Now we compile moses with the correct flags and cmph/boost lib versions

cd /path/to/mosesdecoder/  
./bjam --with-cmph=/Users/urielmandujano/tools/cmph-2.0 --with-boost=/usr/local/boost_1_61_0/ -j4 toolset=clang --with-xmlrpc-c=/usr/local

## Warning
Moses is made for Linux environments and not guaranteed on OSX  
https://www.mail-archive.com/moses-support@mit.edu/msg14530.html

## Running the Moses Translation pivoting
SMT is an inherently lengthy process. To make things easier for me (and hopefully you), it runs in 3 stages. The first time you type python Decoder.py, it will preprocess the data and begin training in the background. Wait until training ends (type ps to see if moses is still working). Once training is complete, run python Decoder.py again to begin tuning, this will take the longest amount of time (again, type ps to check if mert is still working. Expect a long wait time) Finally, once tuning completes, type python Decoder.py again to create the test set, binarize models, and output translations and bleu scores.

###### Notes
1. If after training and tuning you want to toy with your system's translations, use the command  
..* ~/tools/mosesdecoder/bin/moses -minlexr-memory -f fr-en.working/mert-work/moses.ini
2. As a simplifying assumption, I assume the datafile names are of the form something.something.something. For ease of use, best to stick to file name convention I use in the example main()
3. Make sure merge_alignment.py is in mgizapp directory
4. In one of my test runs, the language model could not be built with the given data, giving an Abort Trap: 6 error. Clearing the directory and rerunning fixed the problem. Other issues arise if you aren't using enough data. ~1000 examples should be enough

## TODOs
1. Provide a more user friendly way to provide system the number of CPUs available
2. Allow the user to provide their own directory for mosesdecoder
3. Output bleu scores for individual legs of the translations
