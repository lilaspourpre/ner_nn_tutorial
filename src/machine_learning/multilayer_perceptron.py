# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

class MultilayerPerceptron():
    def __init__(self, input_size, tags, num_neurons):
        self.input_size = input_size
        self.tags = tags
        self.num_neurons = num_neurons
        self.x = tf.placeholder(tf.float32, name='x')
        self.y = tf.placeholder(tf.float32, name='y')
        self.model = self.add_layers(self.num_neurons)
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.model, labels=self.y))
        self.train = tf.contrib.opt.NadamOptimizer().minimize(self.cross_entropy)
        #self.train = tf.train.GradientDescentOptimizer(0.5).minimize(self.cross_entropy)
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)

    def add_layers(self, num_neurons):
        first_layer = self.create_layer(inputs=self.x, input_size=self.input_size,
                                        num_outputs=num_neurons, activation_fn=tf.nn.tanh)
        second_layer = self.create_layer(inputs=first_layer, input_size=int(first_layer.shape[1]),
                                         num_outputs=len(self.tags), activation_fn=None)
        return second_layer

    def create_layer(self, inputs, input_size, num_outputs, activation_fn=tf.nn.relu,
                     weights_initializer=initializers.xavier_initializer(),
                     biases_initializer=tf.zeros_initializer()):
        W = tf.Variable(weights_initializer([input_size, num_outputs]))
        b = tf.Variable(biases_initializer([num_outputs]))
        if activation_fn != None:
            layer = activation_fn(tf.matmul(inputs, W) + b)
        else:
            layer = tf.matmul(inputs, W) + b
        return layer

    def batchnorm_layer(self, tensor, size):
        batch_mean, batch_var = tf.nn.moments(tensor, [0])
        beta = tf.Variable(tf.zeros(size))
        scale = tf.Variable(tf.ones(size))
        return tf.nn.batch_normalization(tensor, batch_mean, batch_var, beta, scale, 0.001)