# -*- coding: utf-8 -*-
from reader import get_documents_with_tags_from
from vector_creator import create_list_of_tagged_vectors
import datetime


# ********************************************************************
#       Main function
# ********************************************************************

def train(model_trainer, feature, documents):
    """
    :param model_trainer:
    :param feature:
    :param morph_analyzer:
    :param path:
    :return:
    """
    list_of_tagged_vectors = create_list_of_tagged_vectors(documents, feature)
    split_lenghts = [length for doc in documents.values() for length in doc.get_sentences_lengths()]
    print('Vectors are created', datetime.datetime.now())
    print(len(list_of_tagged_vectors))
    return model_trainer.batch_train(list_of_tagged_vectors, split_lenghts)
