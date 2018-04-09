# -*- coding: utf-8 -*-
from enitites.features.predicate_feature import PredicateFeature


class CaseFeature(PredicateFeature):
    def __init__(self, predicates=(str.isupper, str.islower, str.istitle)):
        super().__init__(name='case', list_of_predicates=predicates)
