import json
from collections import OrderedDict
from .training_data import TextClass


def get_text_classes_data(data_file, entity_name):
    json_data = json.load(data_file)
    text_classes = OrderedDict()
    for expression in json_data["data"]:
        if "entities" in expression:
            for entity in expression["entities"]:
                if entity["entity"] == entity_name:
                    entity_value = entity["value"]
                    if entity_value not in text_classes:
                        text_classes[entity_value] = TextClass(entity_value, [])
                    text_classes[entity_value].documents.append(expression["text"])
    return text_classes
