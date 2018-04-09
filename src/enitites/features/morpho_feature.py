# -*- coding: utf-8 -*-
from enitites.features.abstract_feature import AbstractFeature


class MorphoFeature(AbstractFeature):
    def __init__(self, name, list_of_strings, tag_function):
        """
        :param name:
        :param list_of_strings:
        :param predicate:
        """
        super().__init__()
        self.name = name
        self.get_tag = tag_function
        self.strings_with_position = {}
        for position in range(len(list_of_strings)):
            self.strings_with_position[list_of_strings[position]] = position

    def compute_vector_for(self, token, document):
        result = [0] * self.get_vector_size()
        parsed_word = document.get_morpho_parsed_tokens()[token.get_id()]
        tag_to_compare = self.get_tag(parsed_word)
        if tag_to_compare in self.strings_with_position:
            result[self.strings_with_position[tag_to_compare]] = 1
        return result

    def get_vector_size(self):
        return len(self.strings_with_position)

    def __repr__(self):
        return self.name
