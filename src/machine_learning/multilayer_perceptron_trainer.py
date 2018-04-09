# -*- coding: utf-8 -*-
import numpy as np
from machine_learning.i_model_trainer import ModelTrainer
from enitites.models.multilayerperceptron_model import MultilayerPerceptronModel


class MultilayerPerceptronTrainer(ModelTrainer):
    def __init__(self, epoch, nn, batch_step):
        super().__init__()
        self.epoch = epoch
        self.nn = nn
        self.batch_step = batch_step

    def train(self, tagged_vectors):
        array_of_vectors = np.array([tagged_vector.get_vector() for tagged_vector in tagged_vectors],
                                    dtype='float32')
        array_of_tags = np.array(
            self.translator([tagged_vector.get_tag() for tagged_vector in tagged_vectors], self.nn.tags),
            dtype='float32')

        for i in range(self.epoch):
            k = int(len(array_of_vectors) / self.batch_step) if len(array_of_vectors) % self.batch_step == 0 \
                else int(len(array_of_vectors) / self.batch_step) + 1
            step = 0
            for j in range(k):
                self.nn.sess.run(self.nn.train, {self.nn.x: array_of_vectors[step:step + self.batch_step],
                                                 self.nn.y: array_of_tags[step:step + self.batch_step]})
                step += self.batch_step

        print("loss: %s" % (self.nn.sess.run([self.nn.cross_entropy], {self.nn.x: array_of_vectors,
                                                                       self.nn.y: array_of_tags})))
        return MultilayerPerceptronModel(self.nn.sess, self.nn.model, self.nn.x, self.nn.tags)
