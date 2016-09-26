# Neural-Network-Machine-Translation
MT using Neural Networks between low-data languages. Continuation of work done in JSALT '15.

Unsupervised translation between language pairs only works if there is a lot of data (documents saying the same thing, in 2 different languages). However, not all language pairs have a lot of data to perform high quality translations. The idea in my project is to use a popular language to link language pairs without a lot of data. For example, perhaps Spanish and French don't have much data, but there is a lot of Spanish- English and English-French data. This project attempts to pivot a translation through the 2 language pairs (Spanish->English->French). The goals in this project are as follows:

1) Develop baseline bleu scores for the pivot translation using standard MT tools

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
git clone https://github.com/moses-smt/mgiza.git
cd mgiza/mgizapp
cmake .
In CMakeList.txt line 39, delete -lrt
make
make install

Now we compile moses with the correct flags and cmph/boost lib versions

cd /path/to/mosesdecoder/
./bjam --with-cmph=/Users/urielmandujano/tools/cmph-2.0 --with-boost=/usr/local/boost_1_61_0/ -j4 toolset=clang --with-xmlrpc-c=/usr/local
