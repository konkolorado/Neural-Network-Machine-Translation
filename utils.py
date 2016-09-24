"""
Utility functions not specific to any class
"""
import os
import sys

import cPickle as pk

def data_exists(folder, ext, lang1_dir, lang2_dir):
    """
    Determines if ext data exists. If so, return True, else False
    Folder is the a str directory to search in
    ext is the str file extension to search for
    lang1_dir, lang2_dir are the paths to the files
    """
    dat1 = make_filename_from_filepath(lang1_dir)
    dat2 = make_filename_from_filepath(lang2_dir)
    if os.path.exists('{}/{}.{}'.format(folder, dat1, ext)) and \
       os.path.exists('{}/{}.{}'.format(folder, dat2, ext)):
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

def write_data(data, filename):
    """
    Writes the given data to a filename in plaint text format
    """
    outstream = open(filename, "w")
    for line in data:
        outstream.write("%s\n" % line)
    outstream.close()

def load_data(filename):
    """
    Given a filename, loads and returns textdata
    """
    instream = open(filename, 'r')
    return [line.strip() for line in instream]

def assert_equal_lens(item1, item2):
    """
    Given two container items, assert that they contain an equal
    number of items and are non empty
    """
    assert len(item1) == len(item2), "Unequal language sizes"
    assert len(item1) and len(item2), "Got language of size 0"

def get_language_extention(filename):
    """
    Given a filename ending in .lang format, returns .lang
    """
    index = filename.rfind('.')
    return filename[index:]

def get_language_root(filename):
    """
    Given a filename ending in .lang format, returns everything before
    .lang
    """
    index = filename.rfind('.')
    return filename[:index]

def directory_name_from_root(file_dir):
    """
    Given a filename, makes a directory name ending in .working
    """
    name_index = get_language_root(file_dir).rfind('.') + 1
    name = get_language_root(file_dir)[name_index:] + ".working"
    return name

def get_full_filename(directory, pattern):
    """
    Given a pattern to search for, will return a matching filename
    in the provided directory
    """
    all_files = os.listdir(directory)
    for f in all_files:
        if pattern in f:
            return f
    return ''
