# -*- coding: utf-8 -*-
from enitites.tagged_vector import TaggedVector


# ********************************************************************
#       Main functions
# ********************************************************************

def create_list_of_tagged_vectors(documents, feature):
    """
    :param documents:
    :param feature:
    :return:
    """
    list_of_tagged_vectors = []

    for document in documents.values():
        for taggedtoken in document.get_tagged_tokens():
            list_of_tagged_vectors.append(__create_tagged_vector_for(taggedtoken, document, feature))
    return list_of_tagged_vectors


# ********************************************************************

def create_dict_of_vectors_for_each_doc(documents, feature):
    """
    :param documents:
    :param feature:
    :return:
    """
    dict_of_tagged_vectors_for_each_doc = {}
    for doc_id, document in documents.items():
        vectors_in_document = create_list_of_tagged_vectors({doc_id: document}, feature)
        dict_of_tagged_vectors_for_each_doc[doc_id] = vectors_in_document
    return dict_of_tagged_vectors_for_each_doc


# --------------------------------------------------------------------

def __create_tagged_vector_for(taggedtoken, document, feature):
    tag = taggedtoken.get_tag()
    vector = feature.compute_vector_for(taggedtoken.get_token(), document)
    return TaggedVector(vector=vector, tag=tag)
