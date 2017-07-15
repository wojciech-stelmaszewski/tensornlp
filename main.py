import numpy as np
import tensornlp as tnlp
import timeit

text_classes = {}
with open("expressions.json", encoding="utf8") as data_file:
    text_classes = tnlp.get_text_classes_data(data_file, "intent_fallback")

coefficients = tnlp.calculate_text_classes_coefficients(text_classes.values())

giga_document = tnlp.create_giga_document(text_classes.values())

dictionary = tnlp.build_dictionary(giga_document)
matrix = tnlp.build_matrix(list(map(lambda class_: class_.create_mega_document(), text_classes.values())), dictionary)

classes_indices = {}
for index, name in enumerate(coefficients):
    classes_indices[name] = index

indices_classes = {}
for index, name in enumerate(coefficients):
    indices_classes[index] = name


classes = np.zeros(len(coefficients))
index = 0
for name in coefficients:
    classes[index] = coefficients[name]
    index += 1


def interpret_result(classification_vector, indices_classes):
    argmax = np.argmax(classification_vector)
    if classification_vector[argmax] > 0.5:
        return {"value": indices_classes[argmax], "confidence": classification_vector[argmax]}
    return {}


def inner_classify(text):
    result = tnlp.classify_text(classes, matrix, dictionary, text)
    start_time = timeit.default_timer()
    elapsed_time = timeit.default_timer() - start_time
    class_result = tnlp.ClassificationResult(text, interpret_result(result, indices_classes), {}, elapsed_time)
    print(class_result)

test_data = []

for test_datum in test_data:
    inner_classify(test_datum)

