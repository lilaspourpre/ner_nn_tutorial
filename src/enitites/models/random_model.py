# -*- coding: utf-8 -*-
import random
from bisect import bisect_right
from enitites.models.model import Model


class RandomModel(Model):
    def __init__(self, list_of_distributed_probabilities):
        super().__init__()
        self.list_of_distributed_probabilities = list_of_distributed_probabilities

    def predict(self, vector):
        random_number = random.uniform(0, 1)
        new_index = bisect_right(self.list_of_distributed_probabilities, (random_number, ''))
        return self.list_of_distributed_probabilities[new_index][1]

    def __repr__(self):
        return 'random_model'

