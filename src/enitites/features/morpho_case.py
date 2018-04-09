# -*- coding: utf-8 -*-
from enitites.features.morpho_feature import MorphoFeature


class MorphoCaseFeature(MorphoFeature):
    CASES = ('nomn', 'gent', 'datv', 'accs', 'ablt', 'loct', 'voct', 'gen2', 'acc2', 'loc2')

    def __init__(self, cases=CASES):
        super().__init__('morphological case', cases, tag_function=self.__select_case)

    @staticmethod
    def __select_case(parsed_word):
        return parsed_word.tag.case
