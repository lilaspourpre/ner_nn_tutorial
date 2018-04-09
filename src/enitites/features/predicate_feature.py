# -*- coding: utf-8 -*-
from enitites.features.abstract_feature import AbstractFeature


class PredicateFeature(AbstractFeature):
    def __init__(self, name, list_of_predicates):
        super().__init__()
        self.name = name
        self.predicates = list_of_predicates

    def compute_vector_for(self, token, document):
        result = []
        for predicate in self.predicates:
            result.append(int(predicate(token.get_text())))
        return result

    def get_vector_size(self):
        return len(self.predicates)

    def __repr__(self):
        return self.name
