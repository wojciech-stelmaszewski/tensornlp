from unidecode import unidecode
from collections import OrderedDict
import math
import numpy as np

UNKNOWN = "UNKNOWN"


def split_text_into_words(text):
    for word in text.split():
        norm_word = unidecode("".join(c for c in word if c.isalnum()).lower())
        if norm_word:
            yield norm_word


def build_dictionary(text):
    dictionary = OrderedDict()
    dictionary[UNKNOWN] = 0
    for word in sorted(split_text_into_words(text)):
        dictionary[word] = 0
    return dictionary


def text_to_vector(text, dictionary):
    vector = np.zeros(len(dictionary.keys()))
    words_indices = list(dictionary)
    for word in split_text_into_words(text):
        word_to_increment = word if word in dictionary else UNKNOWN
        vector[words_indices.index(word_to_increment)] += 1
    return vector


def build_matrix(texts, dictionary):
    row_vectors = []
    for text in texts:
        vector = np.ones(len(dictionary.keys())) + text_to_vector(text, dictionary)
        vector = vector / vector.sum()
        log_vectorized = np.vectorize(math.log)
        vector = log_vectorized(vector)
        row_vectors.append(vector)
    return np.vstack(row_vectors)


def classify(classes, matrix, vector):
    log_vectorized = np.vectorize(math.log)
    exp_vectorized = np.vectorize(math.exp)
    return _normalize_manhattan(exp_vectorized(matrix.dot(vector) + log_vectorized(classes)))


def classify_text(classes, matrix, dictionary, text):
    vector = text_to_vector(text, dictionary)
    return classify(classes, matrix, vector)


def _normalize_manhattan(vector):
    return vector / vector.sum()

