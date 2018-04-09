# -*- coding: utf-8 -*-
from enitites.features.abstract_feature import AbstractFeature


class SpecCharsFeature(AbstractFeature):
    SPEC_CHARS = ('#', '$', '^', '&', '*', '@', '?', '!', '.', '%')

    def __init__(self, spec_chars=SPEC_CHARS):
        super().__init__()
        self.special_chars = {}
        for position in range(len(spec_chars)):
            self.special_chars[spec_chars[position]] = position

    def compute_vector_for(self, token, document):
        result = [0] * self.get_vector_size()
        text_token = token.get_text()
        intersection = set.intersection(set(text_token), set(self.special_chars))
        for symb in intersection:
            result[self.special_chars[symb]] = 1
        return result

    def get_vector_size(self):
        return len(self.special_chars)

    def __repr__(self):
        return 'spec_chars'
