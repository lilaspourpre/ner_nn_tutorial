# -*- coding: utf-8 -*-
from enitites.features.check_element_feature import CheckElementFeature

class PunctFeature(CheckElementFeature):
    SIGNS = (',', '.', '?', '!', ':', '-', '—', '«', '»')

    def __init__(self, punct=SIGNS):
        super().__init__('punctuation', punct, lambda x : x)