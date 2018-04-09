# -*- coding: utf-8 -*-
from collections import Counter
from machine_learning.i_model_trainer import ModelTrainer
from enitites.models.random_model import RandomModel


class RandomModelTrainer(ModelTrainer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def __find_probabilities(list_of_candidates):
        """
        :param list_of_candidates:
        :return: dict_of_distributed_probabilities
        """
        dict_of_candidates_with_counted_tags = Counter(list_of_candidates)
        prev_weight = 0
        list_of_distributed_probabilities = []

        for tag, current_count in dict_of_candidates_with_counted_tags.items():
            current_weight = current_count / len(list_of_candidates)
            list_of_distributed_probabilities.append((prev_weight + current_weight, tag))
            prev_weight += current_weight

        list_of_distributed_probabilities[-1] = (1, list_of_distributed_probabilities[-1][1])
        print(list_of_distributed_probabilities)
        return list_of_distributed_probabilities

    def train(self, tagged_vectors):
        tags = [tagged_vector.get_tag() for tagged_vector in tagged_vectors]
        list_of_distributed_probabilities = self.__find_probabilities(tags)
        return RandomModel(list_of_distributed_probabilities)
