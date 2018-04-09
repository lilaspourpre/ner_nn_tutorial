# -*- coding: utf-8 -*-
from enitites.features.abstract_feature import AbstractFeature


class CheckElementFeature(AbstractFeature):
    def __init__(self, name, elements, text_converter):
        super().__init__()
        self.name = name
        self.text_converter = text_converter
        self.elements_with_position = {}
        for item in elements:
            self.elements_with_position[item] = len(self.elements_with_position)

    def compute_vector_for(self, token, document):
        result = [0] * self.get_vector_size()
        text_token = token.get_text()
        cur_aff = self.text_converter(text_token)
        if cur_aff in self.elements_with_position:
            result[self.elements_with_position[cur_aff]] = 1
        return result

    def get_vector_size(self):
        return len(self.elements_with_position)

    def __repr__(self):
        return self.name
