"""Author: Brandon Trabucco, Copyright 2019
Implements the Best First Module for image captioning."""


import random
import numpy as np
import tensorflow as tf


MY_SEED = 1234567
random.seed(MY_SEED)
np.random.seed(MY_SEED)
tf.set_random_seed(MY_SEED)


from best_first.dataset import make_default


if __name__ == "__main__":

    (vocab, word_embeddings, batch_of_images, batch_of_captions, batch_of_indicators, 
        batch_of_previous, batch_of_pointers, batch_of_words) = make_default()

    for a, b, c, d, e in zip(batch_of_captions, batch_of_indicators, batch_of_previous, 
            batch_of_pointers, batch_of_words):

        print(("\nElement of batch:\n    partial_caption: {0}\n    mask: {1}\n    " + 
            "previous_word: {2}\n    pointer: {3}\n    next_word: {4}").format(
                vocab.id_to_word(a), b, vocab.id_to_word(c), d, vocab.id_to_word(e)))

    
    


