from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from numpy import ndarray
from typing import List, Dict

def load_imdb(num_training: int, num_validation: int, num_test: int,
              num_words : int =10000, max_len : int =None) -> (List[List[int]], List[int]):
    """
    Fetch the IMDB dataset from the web
    """
    # Load IMDB data and use appropriate data types and shapes
    (train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=num_words,
                                                          maxlen=max_len,
                                                          start_char=1, oov_char=2, index_from=3)

    train_y = np.asarray(train_y, dtype=np.int32)
    test_y = np.asarray(test_y, dtype=np.int32)

    # subsample the data
    mask = range(num_training, num_training + num_validation)
    val_x = train_x[mask]
    val_y = train_y[mask]
    mask = range(num_training)
    train_x = train_x[mask]
    train_y = train_y[mask]
    mask = range(num_test)
    test_x = test_x[mask]
    test_y = test_y[mask]

    return train_x, train_y, val_x, val_y, test_x, test_y

def imdb_word_dic() -> (Dict[str, int], Dict[int, str]):
    """
    forms the dictionary of word2index and index2word
    """
    # A dictionary mapping words2index
    word2index = imdb.get_word_index()

    # The first indices are reserved
    word2index = {k: ( v +2) for k, v in word2index.items()}
    word2index["<PAD>"] = 0
    word2index["<START>"] = 1
    word2index["<UNK>"] = 2  # unknown

    # index2word
    index2word = dict([(value, key) for (key, value) in word2index.items()])

    return word2index, index2word


def indices_to_texts(indices: List[int]) -> str:
    """
    convert list of indices to string of text
    """

    return ' '.join(index2word.get(index, "<UNK>") for index in indices)


def texts_to_indices(texts: str) -> List[int]:
    """
    convert string of text to list of integers
    """

    return [word2index.get(word, 2) for word in texts.split()]


def pad_data(train_data, valid_data, test_data,
             word2index: Dict[str, int], max_len: int) -> (ndarray, ndarray):

    train_data = pad_sequences(train_data,
                               value=word2index["<PAD>"],
                               truncating='post',
                               padding='post',
                               maxlen=max_len)

    valid_data = pad_sequences(valid_data,
                               value=word2index["<PAD>"],
                               truncating='post',
                               padding='post',
                               maxlen=max_len)

    test_data = pad_sequences(test_data,
                              value=word2index["<PAD>"],
                              truncating='post',
                              padding='post',
                              maxlen=max_len)

    train_data = np.asarray(train_data, dtype=np.int32)
    valid_data = np.asarray(valid_data, dtype=np.int32)
    test_data = np.asarray(test_data, dtype=np.int32)

    return train_data, valid_data, test_data


def summary_of_data(*args : List[List[int]]) -> None:
    for data in args:
        length = []
        for x in data:
            length.append(len(x))
        print("count: {} maxlen: {} minlen: {} meanlen: {}".format(len(length), max(length), min(length), np.mean(length)))
