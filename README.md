# Statistical Machine Translation via a Pivoting Language

MT using Neural Networks between low-data languages. Continuation of work done in JSALT '15.

This toolkit is designed to facilitate translation between language pairs which may not have much pre-existing parallel training data. It accomplishes this by using the Moses decoded SMT package, freely available at https://github.com/moses-smt/mosesdecoder/tree/RELEASE-2.1.1

Unsupervised translation between language pairs only works if there is a lot of data (documents saying the same thing, in 2 different languages). However, not all language pairs have a lot of data to perform high quality translations. The idea in my project is to use a popular language to link language pairs without a lot of data. For example, perhaps Spanish and French don't have much data, but there is a lot of Spanish- English and English-French data. This project attempts to pivot a translation through the 2 language pairs (Spanish->English->French). The goals in this project are as follows:

1) Develop baseline bleu scores for the pivot translation using standard MT tools

2) Develop a baseline bleu score for performing pivoting using Neural Networks

Before proceeding, it is imperative to note English is not always the most ideal language to use as a pivot.  The biggest advantage to using English is the sheer amount of data available to train and tune a system.  The ideal pivot language however should retain all semantics between languages. For example, in my experiments translating from Spanish to French via English, the system loses gender information.  Both Spanish and French have gender forms for words (una, uno, une, un). The pivot language English however does not. So, we translate Spanish 'una' to English 'one' and into French 'un', when a better system would translate 'una' to '<female> one' to 'une'.  Therefore, we as scientists, must make informed decisions of which pivot language to use and balance the scale between using pivot languages with a lot of parallel training data and using pivot languages which provide the best semantic consistency.

## Getting Started
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
SMT is an inherently lengthy process. To make things easier, this module lets you run the process as a pipeline or use only the functionality you need.

To run the pipeline, simply supply your own options in the config.ini file and run the python3 program pipeline.py.  This pipeline tokenizes the provided data, cleanses it, splits it into training, tuning, and test sets, and matches the test set (for accurate scoring later). After parsing the data, it begins training a model, building language models, tuning the model, filtering the dataset, and testing the result using held out data.

After using the pipeline, you have access not only to Bleu scores for the pivoting translation but you can also translate interactively between languages in each pivot leg and throughout the entire pivot system.

You are welcome to supply your own tokenization and cleansing functions, however, I HIGHLY recommend using the matching and splitting into training, tuning, and testing functionality provided in the parser.  This ensures the data is well suited for the pivoting translation. Additionally, if you find that you have too much training data, there is a subsetting capability provided with the system.

Further usage examples and tips are available in the examples directory.

###### Notes
1. Moses requires a specific filename format during training.  This format is: same file prefix up until  a '.', after which the filename are different. Beyond this, no filename restrictions hold.
2. Make sure merge_alignment.py is in mgizapp directory
3. In one of my test runs, the language model could not be built with the given data, giving an Abort Trap: 6 error. Clearing the working directories and rerunning fixed the problem. Other issues arise if you aren't using enough data. ~1000 examples should be enough.
