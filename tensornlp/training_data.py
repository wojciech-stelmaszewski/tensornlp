from functools import reduce
from collections import OrderedDict

class TextClass(object):
    def __init__(self, name, documents):
        self.name = name
        self.documents = list(documents)

    def create_mega_document(self):
        return " ".join(self.documents)

    def __str__(self):
        return "{0} - {1} documents".format(self.name, len(self.documents))


def create_giga_document(test_classes):
    return " ".join(map(lambda text_class: text_class.create_mega_document(), test_classes))


def calculate_text_classes_coefficients(text_classes):
    coefficients = OrderedDict()
    sum_doc_len = reduce(lambda acc, class_: acc + len(class_.documents), text_classes, 0)
    for text_class in text_classes:
        coefficients[text_class.name] = float(len(text_class.documents))/sum_doc_len
    return coefficients

