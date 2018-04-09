# -*- coding: utf-8 -*-
from abc import abstractmethod


class Model:
    def __init__(self):
        pass

    def split_batch_predict(self, list_of_vectors, split_lengths):
        return self.batch_predict(list_of_vectors)

    def batch_predict(self, list_of_vectors):
        return [self.predict(vector) for vector in list_of_vectors]

    @abstractmethod
    def predict(self, vector):
        """
        :param vector: getting 1 vetor
        :return: tag for the vector
        """
        pass
