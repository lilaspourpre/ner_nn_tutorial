# -*- coding: utf-8 -*-
from enitites.features.abstract_feature import AbstractFeature


class ContextFeature(AbstractFeature):
    def __init__(self, feature: AbstractFeature, offset):
        super().__init__()
        self.feature = feature
        self.offset = offset

    def compute_vector_for(self, token, document):
        tokens_list = document.get_tokens()
        current_index = tokens_list.index(token) + self.offset
        if 0 <= current_index < len(tokens_list):
            current_token = tokens_list[current_index]
            return self.feature.compute_vector_for(current_token, document)
        else:
            return [0] * self.get_vector_size()

    def get_vector_size(self):
        return self.feature.get_vector_size()

    def __repr__(self):
        return repr(self.feature)
