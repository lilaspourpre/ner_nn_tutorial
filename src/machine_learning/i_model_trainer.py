# -*- coding: utf-8 -*-
from abc import abstractmethod


class ModelTrainer:
    def __init__(self):
        pass

    @abstractmethod
    def train(self, tagged_vectors):
        pass

    def translator(self, list_of_all_tags, labels):
        for i in range(len(list_of_all_tags)):
            list_of_labels = [0 if item != list_of_all_tags[i] else 1 for item in labels]
            list_of_all_tags[i:i + 1] = [list_of_labels]
        return list_of_all_tags

    def batch_train(self, tagged_vectors, division):
        return self.train(tagged_vectors)
