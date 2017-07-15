class ClassificationResult(object):
    def __init__(self, text, intent, intent_fallback, elapsed_time):
        self.text = text
        self.intent = intent["value"] if "value" in intent else ""
        self.intent_confidence = \
            str(intent["confidence"]) if "confidence" in intent else ""
        self.intent_fallback = \
            intent_fallback["value"] if "value" in intent_fallback else ""
        self.intent_fallback_confidence = \
            intent_fallback["confidence"] if "confidence" in intent_fallback else ""
        self.elapsed_time = elapsed_time

    def __str__(self):
        return "{0}\t{1}\t{2}\t{3}\t{4}\t{5}".format(self.text, self.intent, self.intent_confidence,
                                                     self.intent_fallback, self.intent_fallback_confidence,
                                                     self.elapsed_time)
