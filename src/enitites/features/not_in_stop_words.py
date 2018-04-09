# -*- coding: utf-8 -*-
from enitites.features.check_in_list_feature import CheckInListFeature


class StopWordsFeature(CheckInListFeature):
    STOP_WORDS = ('ЖКХ', 'ОБЖ', 'ЭВМ', 'БИО')

    def __init__(self, stop_words=STOP_WORDS):
        super().__init__('stop words', stop_words, forward=False)
