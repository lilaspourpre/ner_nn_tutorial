# -*- coding: utf-8 -*-
import numpy as np
from enitites.models.model import Model


class SvmModel(Model):
    def __init__(self, svm):
        super().__init__()
        self.svm = svm

    def predict(self, vector):
        array_of_vectors = np.array([vector])
        return self.svm.predict(array_of_vectors)[0]

    def __repr__(self):
        return 'svm_model'
