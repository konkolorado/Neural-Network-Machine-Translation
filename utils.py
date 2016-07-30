"""
Utility functions not specific to any class
"""
import os
import sys

import cPickle as pk

def data_exists(ext, lang1_dir, lang2_dir):
    """
    Determines if ext data exists. If so, return True, else False
    """
    tokdat1 = make_filename_from_filepath(lang1_dir)
    tokdat2 = make_filename_from_filepath(lang2_dir)
    if os.path.exists('data/{}.{}'.format(tokdat1, ext)) and \
       os.path.exists('data/{}.{}'.format(tokdat2, ext)):
        return True
    return False

def force_print(item):
    """
    Force prints the item to stdout
    """
    sys.stdout.write(str(item))
    sys.stdout.flush()

def make_dir(path):
    """
    Determines if a given path name is a valid directory. If not, makes it
    """
    if not os.path.isdir(path):
        os.makedirs(path)

def make_filename_from_filepath(path):
    """
    Given a path to a file, this function finds the filename and returns
    it
    """
    return os.path.split(path)[1]

def file_exists(filename):
    """
    Returns true if filename exists, else False
    """
    return os.path.isfile(filename)

def pickle_data(data, filename):
    """
    Pickles the given data in the given filename
    """
    outstream = open(filename, 'wb')
    pk.dump(data, outstream)
    outstream.close()

def unpickle_data(filename):
    """
    Given a filename, unpickles and returns the data at that file
    """
    instream = open(filename, 'rb')
    data = pk.load(instream)
    instream.close()
    return data

def assert_equal_lens(item1, item2):
    """
    Given two container items, assert that they contain an equal
    number of items and are non empty
    """
    assert len(item1) == len(item2), "Unequal language sizes"
    assert len(item1) and len(item2), "Got language of size 0"
