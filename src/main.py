# -*- coding: utf-8 -*-
import argparse
from datetime import datetime
import os
import pymorphy2
from gensim.models.fasttext import FastText
from itertools import product
import trainer
import ne_creator
from enitites.features.composite import FeatureComposite
from enitites.features.part_of_speech import POSFeature
from enitites.features.length import LengthFeature
from enitites.features.numbers import NumbersInTokenFeature
from enitites.features.case import CaseFeature
from enitites.features.morpho_case import MorphoCaseFeature
from enitites.features.context_feature import ContextFeature
from enitites.features.special_chars import SpecCharsFeature
from enitites.features.letters import LettersFeature
from enitites.features.df import DFFeature
from enitites.features.position_in_sentence import PositionFeature
from enitites.features.not_in_stop_words import StopWordsFeature
from enitites.features.case_concordance import ConcordCaseFeature
from enitites.features.punctuation import PunctFeature
from enitites.features.prefix_feature import PrefixFeature
from enitites.features.suffix_feature import SuffixFeature
from enitites.features.if_no_lowercase import LowerCaseFeature
from enitites.features.gazetteer import GazetterFeature
from enitites.features.embedding_feature import EmbeddingFeature
from machine_learning.cnn_trainer import CNNTrainer
from machine_learning.majorclass_model_trainer import MajorClassModelTrainer
from machine_learning.multilayer_perceptron import MultilayerPerceptron
from machine_learning.multilayer_perceptron_trainer import MultilayerPerceptronTrainer
from machine_learning.rnn import RNN
from machine_learning.cnn import CNN
from machine_learning.rnn_trainer import RNNTrainer
from machine_learning.random_model_trainer import RandomModelTrainer
from machine_learning.svm_model_trainer import SvmModelTrainer
from reader import get_documents_with_tags_from, get_documents_without_tags_from
import warnings

warnings.filterwarnings('ignore')


# ********************************************************************
#       Main function
# ********************************************************************



def main():
    print(datetime.now())
    args = parse_arguments()
    morph_analyzer = pymorphy2.MorphAnalyzer()
    output_path = os.path.join(args.output_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    embedding_model = get_model_for_embeddings(args.model_path)
    print('Model is ready')
    train_documents = get_documents_with_tags_from(args.trainset_path, morph_analyzer)
    print('Docs are ready for training', datetime.now())
    test_documents = get_documents_without_tags_from(args.testset_path, morph_analyzer)
    print('Docs are ready for testing', datetime.now())
    model_trainer, feature = choose_model(args.algorythm, args.window, train_documents=train_documents,
                                          test_documents=test_documents, ngram_affixes=args.ngram_affixes,
                                          embedding_model=embedding_model)

    train_and_compute_nes_from(model_trainer=model_trainer, feature=feature, train_documents=train_documents,
                               test_documents=test_documents, output_path=output_path)
    print("Testing finished", datetime.now())
    print('Output path: \n {}'.format(output_path))


# --------------------------------------------------------------------

def parse_arguments():
    """
    :return: args (arguments)
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--algorythm",
                        help='"majorclass", "svm", "ml_pc", "lstm", "bilstm" or "random" or "cnn" options are available',
                        required=True)
    parser.add_argument("-w", "--window", help='window size for context', default=2)
    parser.add_argument("-n", "--ngram_affixes", help='number of n-gramns for affixes', default=2)
    parser.add_argument("-t", "--trainset_path", help="path to the trainset files directory")
    parser.add_argument("-s", "--testset_path", help="path to the testset files directory")
    parser.add_argument("-m", "--model_path", help="path to the vector pre-trained model",
                        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data',
                                             'ru.bin'))
    parser.add_argument("-o", "--output_path", help="path to the output files directory",
                        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output'))

    args = parser.parse_args()
    return args


# --------------------------------------------------------------------

def choose_model(method, window, train_documents, test_documents, ngram_affixes, embedding_model):
    """
    :param window:
    :param method: method from argparse
    :return: model trainer + composite
    """
    if method == 'majorclass':
        return MajorClassModelTrainer(), FeatureComposite()
    elif method == 'random':
        return RandomModelTrainer(), FeatureComposite()
    elif method == 'svm':
        feature = get_composite_feature(window, train_documents, ngram_affixes, embedding_model)
        return SvmModelTrainer(kernel='linear'), feature
    elif method == 'ml_pc':
        tags = compute_tags()
        feature = get_composite_feature(window, train_documents, ngram_affixes, embedding_model)
        mp = MultilayerPerceptron(input_size=int(feature.get_vector_size()), tags=tags, num_neurons=100)
        return MultilayerPerceptronTrainer(epoch=100, nn=mp, batch_step=32), feature
    elif 'lstm' in method:
        tags = compute_tags()
        feature = get_composite_feature(window, train_documents, ngram_affixes, embedding_model)
        if 'bi' in method:
            rnn = RNN(input_size=int(feature.get_vector_size()), output_size=len(tags), hidden_size=100, batch_size=8,
                      bilstm=True)
        else:
            rnn = RNN(input_size=int(feature.get_vector_size()), output_size=len(tags), hidden_size=100, batch_size=8)
        return RNNTrainer(epoch=100, nn=rnn, tags=tags), feature
    elif 'cnn' in method:
        tags = compute_tags()
        feature = get_composite_feature(window, train_documents, ngram_affixes, embedding_model)
        cnn = CNN(input_size=int(feature.get_vector_size()), output_size=len(tags), hidden_size=100, batch_size=8)
        return CNNTrainer(epoch=100, nn=cnn, tags=tags), feature
    else:
        raise argparse.ArgumentTypeError(
            'Value has to be "majorclass" or "random" or "svm" or "ml_pc" or "lstm" or "bilstm" or "cnn"')


def get_composite_feature(window, train_documents, ngram_affixes, embedding_model):
    """
    Adding features to composite
    :return: composite (feature storing features)
    """
    list_of_features = [LengthFeature(), NumbersInTokenFeature(), PositionFeature(), DFFeature(), ConcordCaseFeature(),
                        GazetterFeature(), LowerCaseFeature(), SpecCharsFeature(),
                        StopWordsFeature()] #, EmbeddingFeature(embedding_model)]

    list_of_features.append(
        __compute_affixes(PrefixFeature, ngram_affixes, train_documents, end=ngram_affixes))
    list_of_features.append(
        __compute_affixes(SuffixFeature, ngram_affixes, train_documents, start=-ngram_affixes))

    basic_features = [POSFeature(), CaseFeature(), MorphoCaseFeature(), LettersFeature(), PunctFeature()]
    for feature in basic_features:
        for offset in range(-window, window + 1):
            list_of_features.append(ContextFeature(feature, offset))
    composite = FeatureComposite(list_of_features)
    return composite


def compute_tags():
    tags = ["PER", "LOC", "ORG"]
    bilou = ["B", "I", "L", "U"]
    list_of_tags = ["O"]
    list_of_tags += [''.join(i) for i in product(bilou, tags)]
    return list_of_tags


# --------------------------------------------------------------------


def __compute_affixes(feature, ngram_affixes, documents, start=None, end=None):
    set_of_affixes = set()
    for document in documents.values():
        for token in document.get_counter_token_texts().keys():
            set_of_affixes.add(token[start:end])
    return feature(set_of_affixes, ngram_affixes)


# --------------------------------------------------------------------

def get_model_for_embeddings(model_path):
    model = FastText.load_fasttext_format(model_path)
    return model


# --------------------------------------------------------------------
def train_and_compute_nes_from(model_trainer, feature, train_documents, test_documents, output_path):
    """
    :param model_trainer:
    :param feature:
    :param documents:
    :param testset_path:
    :param output_path:
    :param morph_analyzer:
    :return:
    """
    model = trainer.train(model_trainer, feature, train_documents)
    print("Training finished", datetime.now())
    ne_creator.compute_nes(test_documents, feature, model, output_path)


if __name__ == '__main__':
    main()
