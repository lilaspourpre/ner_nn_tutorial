# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from enitites.models.model import Model

class MultilayerPerceptronModel(Model):
    def __init__(self, session, model, x, tags):
        super().__init__()
        self.session = session
        self.x = x
        self.model = model
        self.tags = tags

    def batch_predict(self, list_of_vectors):
        array_of_vectors = np.array(list_of_vectors, dtype='float32')
        prediction = tf.argmax(self.model,1)
        index_list = prediction.eval(feed_dict={self.x: array_of_vectors}, session=self.session)
        return [self.tags[i] for i in index_list]

    def __repr__(self):
        return 'mlperc_model'
