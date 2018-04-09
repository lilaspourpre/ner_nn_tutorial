# -*- coding: utf-8 -*-
from collections import Counter

import nltk
import pymorphy2
from nltk.tokenize import sent_tokenize

class Document:
    def __init__(self, list_of_tagged_tokens, morph_analyzer=pymorphy2.MorphAnalyzer()):
        self.__list_of_tagged_tokens = list_of_tagged_tokens
        self.__list_of_tokens = self.__compute_tokens()
        self.__dict_of_parsed_tokens = self.__compute_morpho_parsed_tokens(morph_analyzer)
        self.__list_of_sentences = self.__compute_sentences()
        self.__list_of_sentences_length = self.__compute_sentences_lengths()
        self.__list_of_pos_sentences = self.__compute_pos_sentences()
        self.__index_by_tokens = self.__compute_index_by_tokens()
        self.__counter_token_texts = self.__compute_counter_token_texts()

    def get_tagged_tokens(self):
        return self.__list_of_tagged_tokens

    def get_tokens(self):
        return self.__list_of_tokens

    def get_morpho_parsed_tokens(self):
        return self.__dict_of_parsed_tokens

    def get_pos_sentences(self):
        return self.__list_of_pos_sentences

    def get_sentences(self):
        return self.__list_of_sentences

    def get_sentences_lengths(self):
        return self.__list_of_sentences_length

    def get_counter_token_texts(self):
        return self.__counter_token_texts

    def get_index_by_token(self, token):
        return self.__index_by_tokens[token]

    def get_id_by_index(self, index):
        return self.__list_of_tokens[index].get_id()

    def get_size(self):
        return len(self.__list_of_tokens)

    def __compute_tokens(self):
        tokens = []
        for tagged_token in self.__list_of_tagged_tokens:
            tokens.append(tagged_token.get_token())
        return tokens

    def __compute_morpho_parsed_tokens(self, morph_analyzer):
        dict_of_parsed_words = {}
        for token in self.__list_of_tokens:
            parsed_word = morph_analyzer.parse(token.get_text())[0]
            dict_of_parsed_words[token.get_id()] = parsed_word
        return dict_of_parsed_words

    def __compute_index_by_tokens(self):
        index_by_tokens = {}
        tokens = self.get_tokens()
        for i in range(len(tokens)):
            index_by_tokens[tokens[i]] = i
        return index_by_tokens

    def __compute_counter_token_texts(self):
        list_of_text_tokens = [token.get_text() for token in self.get_tokens()]
        return Counter(list_of_text_tokens)

    # XXX actually, this info is within input file. No need to recompute it, especially with nltk
    def __compute_sentences(self):
        list_of_all_words = [t.get_text() for t in self.get_tokens()]
        sent_detector = nltk.data.load('../../data/nltk_data/tokenizers/punkt/russian.pickle')
        sen = sent_detector.tokenize(' '.join(list_of_all_words))
        return sen

    def __compute_pos_sentences(self):
        # https://github.com/mhq/train_punkt/blob/master/russian.pickle
        list_of_all_ids = [t.get_id() for t in self.get_tokens()]
        dict_of_pos_in_sentences = {}
        index = 0
        for sent in self.__list_of_sentences:
            sent_split = sent.split()
            for i in range(len(sent_split) - 1):
                dict_of_pos_in_sentences[list_of_all_ids[index + i]] = i
            # XXX bad idea: not logical and tightly coupled with position feature implementation
            dict_of_pos_in_sentences[list_of_all_ids[index + len(sent_split) - 1]] = -1
            index += len(sent_split)
        return dict_of_pos_in_sentences

    def __compute_sentences_lengths(self):
        lengths_list = [len(sentence.split()) for sentence in self.__list_of_sentences]
        return lengths_list
