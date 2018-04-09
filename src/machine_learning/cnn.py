# -*- coding: utf-8 -*-
import tensorflow as tf


class CNN():
    def __init__(self, input_size, output_size, hidden_size, batch_size, filter_sizes=(3, 4, 5)):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.filter_sizes = filter_sizes

        self.x = tf.placeholder(tf.float32, [None, None, self.input_size], name='x')
        self.x_new = tf.expand_dims(self.x, -1)
        self.y = tf.placeholder(tf.float32, [None, None, self.output_size], name='y')
        self.seqlen = tf.placeholder(tf.int32, [None])

        output_layer = self.__add_convolution_layer(self.x, self.input_size, self.hidden_size)

        self.outputs = tf.layers.dense(output_layer, self.output_size, activation=tf.tanh,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.loss = self.__add_loss(self.outputs)
        self.train = self.__add_update_step(self.loss)

        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)

    def __add_convolution_layer(self, input_tensor, input_size, output_size):
        input_tensor_expand = tf.expand_dims(input_tensor, -1)
        conv_outputs = []

        for i in range(len(self.filter_sizes)):
            filter_size = self.filter_sizes[i]
            conv_1 = self.__apply_convolution(filter_size, input_tensor_expand, input_size, output_size)
            conv_2 = self.__apply_convolution(filter_size, input_tensor_expand, input_size, output_size)
            result = conv_1 * tf.sigmoid(conv_2)
            #max_pool_outputs = tf.nn.max_pool(result, [1, filter_size, 1, 1], [1, 1, 1, 1], 'SAME')
            conv_outputs.append(result)

        conv_first = tf.concat(conv_outputs, 3)
        conv_first = tf.squeeze(conv_first, [2])
        return conv_first

    def __apply_convolution(self, filter_size, input_tensor, input_size, output_size):
        filter = tf.Variable(tf.random_normal([filter_size, input_size, 1, output_size], stddev=0.1))
        bias = tf.Variable(tf.constant(0.1, shape=[output_size]), name="b")
        conv_outputs = tf.nn.conv2d(input_tensor, filter, [1, 1, input_size, 1], 'SAME')
        conv_outputs = tf.tanh(tf.nn.bias_add(conv_outputs, bias))
        return conv_outputs

    def __add_loss(self, logits):
        mask = tf.sequence_mask(
            self.seqlen,
            dtype=tf.float32)

        losses = tf.squeeze(
            tf.losses.cosine_distance(
                tf.nn.l2_normalize(self.y, -1),
                logits,
                axis=-1,
                weights=1.0,
                reduction=tf.losses.Reduction.NONE),
            [-1])

        loss = tf.reduce_sum(losses * mask) / tf.cast(tf.reduce_sum(self.output_size), tf.float32)
        return loss

    def __add_update_step(self, loss):
        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, self.gradients_norm = tf.clip_by_global_norm(gradients, 1)

        optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=0.001)
        # optimizer = tf.train.AdamOptimizer(learning_rate)
        update_step = optimizer.apply_gradients(zip(clipped_gradients, params))
        return update_step
