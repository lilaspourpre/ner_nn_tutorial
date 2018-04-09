# -*- coding: utf-8 -*-
from collections import Counter
from machine_learning.i_model_trainer import ModelTrainer
from enitites.models.majorclass_model import MajorClassModel


class MajorClassModelTrainer(ModelTrainer):
    def __init__(self):
        super().__init__()

    def __find_majority_class(self, list_of_candidates):
        """
        :param list_of_candidates:
        :return:
        """
        counted_tags = Counter(list_of_candidates)
        return counted_tags.most_common(1)[0][0]

    def train(self, tagged_vectors):
        tags = [tagged_vector.get_tag() for tagged_vector in tagged_vectors]
        major_class = self.__find_majority_class(tags)
        return MajorClassModel(major_class)
