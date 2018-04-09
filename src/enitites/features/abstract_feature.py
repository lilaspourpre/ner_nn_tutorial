# -*- coding: utf-8 -*-
from abc import abstractmethod


class AbstractFeature:
    def __init__(self):
        pass

    @abstractmethod
    def compute_vector_for(self, token, document):
        """
        :param token: tokenObject
        :param document: documentObject
        :return: vector
        """
        pass

    @abstractmethod
    def get_vector_size(self):
        pass
