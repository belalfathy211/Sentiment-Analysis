import json
import re
from collections import Counter


class Vocabulary:
    def __init__(self, min_freq=5):
        self.min_freq = min_freq
        self.itos = {0: "<PAD>", 1: "<UNK>", 2: "<START>", 3: "<END>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        text = text.lower()
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return text.split()

    def build_vocabulary(self, tokenized_reviews):
        frequencies = Counter()
        for tokens in tokenized_reviews:
            frequencies.update(tokens)

        idx = 4
        for word, count in frequencies.items():
            if count >= self.min_freq:
                self.itos[idx] = word
                self.stoi[word] = idx
                idx = idx + 1

    def numericalize(self, tokens):
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokens]

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"stoi": self.stoi, "itos": self.itos}, f)
