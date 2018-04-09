# -*- coding: utf-8 -*-
from enitites.features.check_in_list_feature import CheckInListFeature


class GazetterFeature(CheckInListFeature):
    GAZETTER = ('Мария', 'Владимир', 'Ольга', 'Степан', 'России')

    def __init__(self, gazetteer=GAZETTER):
        super().__init__('gazetter', gazetteer)
