"""Author: Brandon Trabucco, Copyright 2019
Implements the Best First Module for image captioning."""


from collections import namedtuple


def make_vocab_from_corpus(corpus):

    tokens, unique, counts = Vocab._tokenize(corpus), [], {}
    for t in tokens:
        if t not in unique:
            unique.append(t)
            counts[t] = 0
        counts[t] = counts[t] + 1
    unique = list(sorted(unique, key=lambda x: -counts[x]))
    return tokens, Vocab(unique, Config("<?>", "<S>", "</S>"))


class Config(namedtuple(
        "Config", [ "unk_word", "start_word", "end_word" ])):

    pass


class Vocab(object):

    @staticmethod
    def _tokenize(sentence):

        sentence = sentence.lower().replace(";", " ; ").replace("\"", " \" ")
        sentence = sentence.replace("(", " ( ").replace(".", " . ").replace("?", " ? ")
        sentence = sentence.replace(",", " , ").replace(":", " : ").replace(")", " ) ")
        sentence = sentence.replace("'", " ' ").replace("!", " ! ").replace("-", " - ")
        sentence = sentence.replace(";", " ; ").replace("  ", " ").replace("  ", " ")
        return sentence.strip().split(" ")


    def __init__(self, word_list, config):

        if config.unk_word not in word_list:
            word_list.append(config.unk_word)
        if config.start_word not in word_list:
            word_list.append(config.start_word)
        if config.end_word not in word_list:
            word_list.append(config.end_word)

        self.vocab = { word : i for i, word in enumerate(word_list) }
        self.reverse_vocab = word_list
        self.unk_id = self.vocab[config.unk_word]
        self.start_id = self.vocab[config.start_word]
        self.end_id = self.vocab[config.end_word]
        self.size = len(self.reverse_vocab)


    def word_to_id(self, word):

        if isinstance(word, list):
            return [self.word_to_id(w) for w in word]
        if word not in self.vocab:
            return self.unk_id
        return self.vocab[word]


    def id_to_word(self, index):

        if isinstance(index, list):
            return [self.id_to_word(i) for i in index]
        if index < 0 or index >= len(self.reverse_vocab):
            return self.reverse_vocab[self.unk_id]
        return self.reverse_vocab[index]


    def tokenize(self, sentence):

        return Vocab._tokenize(sentence)