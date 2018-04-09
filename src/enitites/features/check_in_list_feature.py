# -*- coding: utf-8 -*-
from enitites.features.abstract_feature import AbstractFeature


class CheckInListFeature(AbstractFeature):
    def __init__(self, name, set_of_strings, forward=True):
        """
        :param name:
        :param set_of_strings:
        :param forward:
        """
        super().__init__()
        self.name = name
        self.forward = forward
        self.set_of_strings = set(set_of_strings)

    def compute_vector_for(self, token, document):
        text_token = token.get_text()
        return self.__check(text_token in self.set_of_strings)

    def __check(self, result):
        return [int(self.forward == result)]

    def get_vector_size(self):
        return 1

    def __repr__(self):
        return self.name
