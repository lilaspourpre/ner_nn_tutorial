# -*- coding: utf-8 -*-
import numpy as np
from machine_learning.i_model_trainer import ModelTrainer
from enitites.models.svm_model import SvmModel
from sklearn import svm, model_selection


class SvmModelTrainer(ModelTrainer):
    def __init__(self, kernel=None):
        super().__init__()
        self.kernel = kernel
        if kernel == 'linear':
            self.svm = svm.LinearSVC(C=1, class_weight=None, dual=True, fit_intercept=True,
                        intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr',
                        penalty='l2', random_state=None, tol=0.0001, verbose=0)
        else:
            self.svm = svm.SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
                               decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                               max_iter=-1, probability=False, random_state=None, shrinking=True,
                               tol=0.001, verbose=False)

    def train(self, tagged_vectors):
        # print(tagged_vectors)
        array_of_vectors = np.array([tagged_vector.get_vector() for tagged_vector in tagged_vectors])
        array_of_tags = np.array([tagged_vector.get_tag() for tagged_vector in tagged_vectors])
        #best_estimator = self.do_grid_search(array_of_vectors, array_of_tags)
        #self.svm = best_estimator
        self.svm.fit(array_of_vectors, array_of_tags)
        return SvmModel(self.svm)

    def do_grid_search(self, array_of_vectors, array_of_tags):
        if self.kernel == 'linear':
            parameters = {'penalty': ('l1', 'l2'), 'loss': ('hinge', 'squared_hinge'),
                          'multi_class': ('ovr', 'crammer_singer'), 'C': [0.01, 0.1, 1, 10]}
        else:
            parameters={ 'C' : [0.01, 0.1, 1, 5, 10], 'kernel': ['rbf', 'linear', 'poly'],
                         'degree': np.arange( 0, 100+0, 1 ).tolist(),
                         'gamma': np.arange( 0.0, 10.0+0.0, 0.1 ).tolist(),
                         'decision_function_shape' : ['ovo', 'ovr']}
        clf = model_selection.GridSearchCV(self.svm, parameters, error_score=0.0)
        clf.fit(array_of_vectors, array_of_tags)
        print(clf.best_estimator_)
        return clf.best_estimator_
