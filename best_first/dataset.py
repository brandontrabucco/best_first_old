"""Author: Brandon Trabucco, Copyright 2019
Implements the Best First Module for image captioning."""


import numpy as np
from best_first.vocab import make_vocab_from_corpus


def make_default():

    caption = "a black and white spotted cat sleeping on a sofa cushion ."
    tokens, vocab = make_vocab_from_corpus(caption)
    word_embeddings = np.random.normal(0, 1, [vocab.size, 300]).astype(np.float32)
    image = np.random.normal(0, 1, [2048])
    ids = vocab.word_to_id(tokens)

    batch_of_words = ids + [vocab.end_id]
    batch_of_previous = [vocab.start_id] + batch_of_words[:-1]
    batch_of_captions = [[vocab.start_id] + batch_of_words[:i] + [
        vocab.end_id] for i in range(len(batch_of_words))]
    batch_of_lengths = [len(x) for x in batch_of_captions]
    batch_of_pointers = np.arange(len(batch_of_words), dtype=np.int32)
    batch_of_images = np.stack([image for _ in range(len(batch_of_words))])

    max_length = max(batch_of_lengths)
    batch_of_indicators = [[1.0] * (x) + [0.0] * (max_length - 
        x) for x in batch_of_lengths]
    for x, y in zip(batch_of_captions, batch_of_lengths):
        x.extend([0] * (max_length - y))

    return (vocab, word_embeddings, batch_of_images, batch_of_captions, 
        batch_of_indicators, batch_of_previous, batch_of_pointers, batch_of_words)
