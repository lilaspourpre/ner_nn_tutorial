# -*- coding: utf-8 -*-
import re
from enitites.features.predicate_feature import PredicateFeature


class LettersFeature(PredicateFeature):
    def __init__(self, reg_exps=(re.compile(u'[a-zA-Z]+'), re.compile(u'[а-яА-Я]+'))):
        predicates = [self.__create_search_predicate(reg) for reg in reg_exps]
        super().__init__(name='letters_type', list_of_predicates=predicates)

    @staticmethod
    def __create_search_predicate(reg):
        return lambda x: 0 if reg.search(x) is None else 1
