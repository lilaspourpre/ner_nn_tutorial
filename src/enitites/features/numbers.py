# -*- coding: utf-8 -*-
from enitites.features.predicate_feature import PredicateFeature  # XXX src is still here


class NumbersInTokenFeature(PredicateFeature):
    def __init__(self, predicates=(str.isalpha, str.isdigit, str.isalnum)):
        super().__init__('numbers_in_token', predicates)
