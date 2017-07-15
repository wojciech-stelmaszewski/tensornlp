import tensornlp as tnlp

text_classes = {}
with open("expressions.json", encoding="utf8") as data_file:
    text_classes = tnlp.get_text_classes_data(data_file, "intent")

coefficients = tnlp.calculate_text_classes_coefficients(text_classes.values())

for text_class in text_classes.values():
    print(text_class)
