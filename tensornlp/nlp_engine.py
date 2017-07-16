from .naive_bayes import *
from .expressions import *
from .training_data import *
from wit import Wit
from time import sleep
import timeit
import numpy as np


class TensorNLPResult(object):

    def __init__(self, text, entity_name, entity_value, confidence, elapsed_time):
        self.text = text
        self.entity_name = entity_name
        self.entity_value = entity_value
        self.confidence = confidence
        self.elapsed_time = elapsed_time

    def __str__(self):
        if self.entity_value:
            return "{0}:\t{1}\t[{2:.4f}]\t-\t{3:.4f} s".format(self.entity_name, self.entity_value,
                                                               self.confidence, self.elapsed_time)
        else:
            return "{0}:\t\t\t-\t{1:.4f} s".format(self.entity_name, self.elapsed_time)


class TensorNLPEngine(object):

    def __init__(self, expressions_file_path, entity_name):
        self.entity_name = entity_name
        with open(expressions_file_path, encoding="utf8") as data_file:
            text_classes = get_text_classes_data(data_file, entity_name)

        coefficients = calculate_text_classes_coefficients(text_classes.values())

        giga_document = create_giga_document(text_classes.values())

        self.dictionary = build_dictionary(giga_document)
        self.matrix = build_matrix(list(map(lambda class_: class_.create_mega_document(), text_classes.values())),
                                   self.dictionary)

        self.classes_indices = {}
        for index, name in enumerate(coefficients):
            self.classes_indices[name] = index

        self.indices_classes = {}
        for index, name in enumerate(coefficients):
            self.indices_classes[index] = name

        self.classes = np.zeros(len(coefficients))
        index = 0
        for name in coefficients:
            self.classes[index] = coefficients[name]
            index += 1

    def classify(self, text):
        start_time = timeit.default_timer()
        result = classify_text(self.classes, self.matrix, self.dictionary, text)
        elapsed_time = timeit.default_timer() - start_time

        argmax = np.argmax(result)
        entity_result = {} if result[argmax] < 0.5 else \
            {"value": self.indices_classes[argmax], "confidence": result[argmax]}

        class_result = TensorNLPResult(text, self.entity_name,
                                       entity_result["value"] if "value" in entity_result else "",
                                       entity_result["confidence"] if "confidence" in entity_result else "",
                                       elapsed_time)
        return class_result


class WitNLPResult(object):

    def __init__(self, text, intent, intent_fallback, elapsed_time):
        self.text = text
        self.intent = intent["value"] if "value" in intent else ""
        self.intent_confidence = \
            intent["confidence"] if "confidence" in intent else ""
        self.intent_fallback = \
            intent_fallback["value"] if "value" in intent_fallback else ""
        self.intent_fallback_confidence = \
            intent_fallback["confidence"] if "confidence" in intent_fallback else ""
        self.elapsed_time = elapsed_time

    def __str__(self):
        return "intent:\t{1}\t[{2:.4f}]\n" \
               "intent_fallback:\t{3}\t[{4:.4f}]\n" \
               "{5:.4f} s".format(self.text, self.intent, self.intent_confidence,
                               self.intent_fallback, self.intent_fallback_confidence,
                               self.elapsed_time)


class WitNLPEngine(object):

    def __init__(self, access_token):
        self.wit_client = Wit(access_token)
        self.sleep_time = 5

    def classify(self, text):
        sleep(self.sleep_time)
        start_time = timeit.default_timer()
        result = self.wit_client.message(text)
        elapsed_time = timeit.default_timer() - start_time

        intent = {}
        if "intent" in result["entities"]:
            intent = sorted(result["entities"]["intent"],
                            key=lambda entity: entity["confidence"])[0]

        intent_fallback = {}
        if "intent_fallback" in result["entities"]:
            intent_fallback = sorted(result["entities"]["intent_fallback"],
                                     key=lambda entity: entity["confidence"])[0]

        return WitNLPResult(text, intent, intent_fallback, elapsed_time)


