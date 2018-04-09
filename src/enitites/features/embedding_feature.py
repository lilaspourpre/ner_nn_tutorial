# -*- coding: utf-8 -*-
from enitites.features.abstract_feature import AbstractFeature
import numpy as np
from sklearn.manifold import TSNE

class EmbeddingFeature(AbstractFeature):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def compute_vector_for(self, token, document):
        text_token = token.get_text()
        try:
            vector = self.model[text_token]
            return list(vector)
        except KeyError:
            return [0] * self.get_vector_size()

    def get_vector_size(self):
        return self.model.vector_size

    def __repr__(self):
        return 'embedding feature'