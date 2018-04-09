# -*- coding: utf-8 -*-
import tensorflow as tf


class RNN():
    def __init__(self, input_size, output_size, hidden_size, batch_size, bilstm=False):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.x = tf.placeholder(tf.float32, [None, None, self.input_size], name='x')
        self.y = tf.placeholder(tf.float32, [None, None, self.output_size], name='y')
        self.seqlen = tf.placeholder(tf.int32, [None])
        self.fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
        if bilstm:
            self.bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
            self.mid_outputs, state = tf.nn.bidirectional_dynamic_rnn(self.fw_cell, self.bw_cell, self.x,
                                                                      sequence_length=self.seqlen,
                                                                      dtype=tf.float32)
            self.mid_outputs = tf.concat(self.mid_outputs, 2)
        else:
            self.mid_outputs, state = tf.nn.dynamic_rnn(self.fw_cell, self.x, sequence_length=self.seqlen,
                                                    dtype=tf.float32)
        self.outputs = tf.contrib.layers.fully_connected(self.mid_outputs, self.output_size, activation_fn=None)
        #self.outputs = tf.contrib.layers.fully_connected(self.mid_outputs[-1], self.output_size, activation_fn=None)

        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.outputs,
                                                    labels=self.y))
        self.train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.cross_entropy)

        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)
