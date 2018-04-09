# -*- coding: utf-8 -*-
import os
import codecs
import csv
from datetime import datetime

from bilou import from_bilou
from reader import get_documents_without_tags_from
from vector_creator import create_dict_of_vectors_for_each_doc


# ********************************************************************
#       Main function
# ********************************************************************

def compute_nes(documents, feature, model, output_path):
    """
    :param morph_analyzer: 
    :param output_path: 
    :param testset_path:
    :param feature:
    :param model:
    :return:
    """
    dict_of_docs_with_vectors = create_dict_of_vectors_for_each_doc(documents, feature)
    for document_name, untagged_vectors_list in dict_of_docs_with_vectors.items():
        list_of_vectors = [untagged_vector.get_vector() for untagged_vector in untagged_vectors_list]
        split_lenghts = documents[document_name].get_sentences_lengths()
        ne_list = __define_nes(model, list_of_vectors, documents[document_name].get_tokens(), split_lenghts)
        __write_to_file(ne_list, document_name, output_path)


# --------------------------------------------------------------------

def __define_nes(model, vectors_list, tokens, split_lengths):
    list_of_tags = model.split_batch_predict(vectors_list, split_lengths)
    return from_bilou.untag(list_of_tags=list_of_tags, list_of_tokens=tokens)


# --------------------------------------------------------------------

def __write_to_file(nes, filename, path):
    """
    :param nes:
    :param filename:
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)

    with codecs.open(os.path.join(path, os.path.basename(filename + ".task1")), 'w', encoding='utf-8') as f:
        for ne in nes:
            writer = csv.writer(f, delimiter=' ')
            writer.writerow(ne)
