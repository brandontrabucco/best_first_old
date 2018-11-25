"""Author: Brandon Trabucco, Copyright 2019
Implements the Best First Module for image captioning."""


import random
import numpy as np
import tensorflow as tf


MY_SEED = 1234567
random.seed(MY_SEED)
np.random.seed(MY_SEED)
tf.set_random_seed(MY_SEED)


from best_first.module import BestFirstModule
from best_first.dataset import make_default


if __name__ == "__main__":

    (vocab, word_embeddings, batch_of_images, batch_of_captions, batch_of_indicators, 
        batch_of_previous, batch_of_pointers, batch_of_words) = make_default()

    single_image = batch_of_images[0, :]

    with tf.Graph().as_default():

        batch_of_images = tf.placeholder(tf.float32, shape=[1, 2048])
        batch_of_captions = tf.placeholder(tf.int32, shape=[1, None])
        batch_of_previous = tf.placeholder(tf.int32, shape=[1])

        module = BestFirstModule(word_embeddings)
        pointer_logits, word_logits = module(batch_of_images, batch_of_captions, 
            batch_of_previous)
        global_step = tf.train.get_or_create_global_step()

        saver = tf.train.Saver(var_list=module.variables + [global_step])

        with tf.Session() as sess:

            saver.restore(sess, "./ckpts/model.ckpt")
            current_caption = [vocab.start_id, vocab.end_id]
            previous_id = vocab.start_id

            while len(current_caption) < 99:

                p, w = sess.run([pointer_logits, word_logits], feed_dict={
                    batch_of_images: [single_image],
                    batch_of_captions: [current_caption],
                    batch_of_previous: [previous_id] })
                index = np.argmax(p.flatten())
                previous_id = np.argmax(w.flatten())
                if previous_id == vocab.end_id:
                    break
                current_caption.insert(index + 1, previous_id)

            print(vocab.id_to_word(current_caption))
        




    
    


