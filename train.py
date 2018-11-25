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

    with tf.Graph().as_default():

        batch_of_images = tf.constant(batch_of_images, dtype=tf.float32)
        batch_of_captions = tf.constant(batch_of_captions, dtype=tf.int32)
        batch_of_indicators = tf.constant(batch_of_indicators, dtype=tf.float32)
        batch_of_previous = tf.constant(batch_of_previous, dtype=tf.int32)
        batch_of_pointers = tf.constant(batch_of_pointers, dtype=tf.int32)
        batch_of_words = tf.constant(batch_of_words, dtype=tf.int32)

        module = BestFirstModule(word_embeddings)
        pointer_logits, word_logits = module(batch_of_images, batch_of_captions, 
            batch_of_previous, indicators=batch_of_indicators, pointer_ids=batch_of_pointers)

        tf.losses.sparse_softmax_cross_entropy(batch_of_pointers, pointer_logits)
        tf.losses.sparse_softmax_cross_entropy(batch_of_words, word_logits)

        loss = tf.losses.get_total_loss()
        global_step = tf.train.get_or_create_global_step()

        learning_rate = tf.train.exponential_decay(0.01, global_step, 100, 0.99, staircase=False)
        training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
            loss, global_step=global_step, var_list=module.variables)

        saver = tf.train.Saver(var_list=module.variables + [global_step])

        with tf.Session() as sess:

            sess.run(tf.variables_initializer(module.variables + [global_step]))
            for i in range(10000):
                sess.run(training_step)
                print("Training Iteration {0:05d} | loss: {1:.5f} ".format(
                    *sess.run([global_step, loss])))

            saver.save(sess, "./ckpts/model.ckpt")

        




    
    


