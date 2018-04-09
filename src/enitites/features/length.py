# -*- coding: utf-8 -*-
from enitites.features.abstract_feature import AbstractFeature


class LengthFeature(AbstractFeature):
    def __init__(self):
        super().__init__()

    def compute_vector_for(self, token, document):
        return [len(token.get_text())]

    def get_vector_size(self):
        return 1

    def __repr__(self):
        return 'length'
