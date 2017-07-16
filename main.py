import tensornlp as tnlp
from collections import OrderedDict
import shlex

PROMPT = "|> "

nlp_engines = OrderedDict()


def add_engine(engine, engine_name, *args):
    if engine == "wit":
        print("Added WitNLPEngine with name \"{0}\" and parameters \"{1}\"".format(engine_name, args[0]))
        wit_nlp_engine = tnlp.WitNLPEngine(args[0][0])
        nlp_engines[engine_name] = wit_nlp_engine
    elif engine == "tensornlp":
        print("Added TensorNLPEngine with name \"{0}\" and parameters \"{1}\"".format(engine_name, args[0]))
        tensor_nlp_engine = tnlp.TensorNLPEngine(args[0][0], args[0][1])
        nlp_engines[engine_name] = tensor_nlp_engine
    else:
        print("Unknown NLP engine \"{0}\"!".format(engine))

while True:
    user_input = input(PROMPT)

    command = shlex.split(user_input)

    if command[0].lower() == "ping":
        print("pong")
        continue

    if command[0].lower() == "add":
        add_engine(command[1].lower(), command[2], command[3:])
        continue

    if command[0].lower() == "exit":
        break

    for nlp_engine in nlp_engines:
        print("{0}\n{1}\n".format(nlp_engine, nlp_engines[nlp_engine].classify(user_input)))
