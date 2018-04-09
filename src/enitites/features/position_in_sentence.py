# -*- coding: utf-8 -*-
from enitites.features.abstract_feature import AbstractFeature


class PositionFeature(AbstractFeature):
    def __init__(self):
        super().__init__()
        self.indexes = [0, -1] # XXX idea with -1 is bad - see comment in document

    def compute_vector_for(self, token, document):
        result = [1, 1] # XXX acceptable, but strange - usually we init with zeros
        dict_of_pos_in_sentences = document.get_pos_sentences()
        for i in self.indexes:
            if dict_of_pos_in_sentences[token.get_id()] == i:
                result[i] = 0
        return result

    def get_vector_size(self):
        return 2

    def __repr__(self):
        return 'position in sentence'
