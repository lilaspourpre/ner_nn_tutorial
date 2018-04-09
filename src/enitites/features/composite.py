# -*- coding: utf-8 -*-
from enitites.features.abstract_feature import AbstractFeature


class FeatureComposite(AbstractFeature):
    def __init__(self, feature_list=()):
        super().__init__()
        self.feature_list = feature_list

    def compute_vector_for(self, token, document):
        final_vector = []
        for feature in self.feature_list:
            final_vector.extend(feature.compute_vector_for(token, document))
        return final_vector

    def get_vector_size(self):
        return sum([i.get_vector_size() for i in self.feature_list])

    def __repr__(self):
        return "<<" + ', '.join([repr(i) for i in self.feature_list]) + ">>"
