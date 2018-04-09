# -*- coding: utf-8 -*-
from enitites.models.model import Model


class MajorClassModel(Model):
    def __init__(self, major_class):
        super().__init__()
        self.major_class = major_class

    def predict(self, vector):
        return self.major_class

    def __repr__(self):
        return 'majorclass_model'
