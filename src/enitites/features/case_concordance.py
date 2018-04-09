# -*- coding: utf-8 -*-
from enitites.features.abstract_feature import AbstractFeature


class ConcordCaseFeature(AbstractFeature):
    CASES = ('nomn', 'gent', 'datv', 'accs', 'ablt', 'loct', 'voct', 'gen2', 'acc2', 'loc2')

    def __init__(self, cases_to_detect=CASES):
        super().__init__()
        self.case_to_position = {}
        for position in range(len(cases_to_detect)):
            self.case_to_position[cases_to_detect[position]] = position

    def compute_vector_for(self, token, document):
        parsed_word = document.get_morpho_parsed_tokens()[token.get_id()]
        current_case = str(parsed_word.tag.case)
        if current_case is None:
            return [0, 0]
        else:
            index = document.get_index_by_token(token)
            return [self.__compare_cases(current_case, index - 1, document),
                    self.__compare_cases(current_case, index + 1, document)]

    def __compare_cases(self, current_case, other_index, document):
        if other_index != -1 and other_index != len(document.get_tokens()):
            parsed_word = document.get_morpho_parsed_tokens()[document.get_id_by_index(other_index)]
            new_case = str(parsed_word.tag.case)
            if new_case is None:
                return 0
            else:
                return int(current_case == new_case)
        else:
            return 0

    def get_vector_size(self):
        return 2

    def __repr__(self):
        return 'morphological case'
