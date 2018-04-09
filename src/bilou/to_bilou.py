# -*- coding: utf-8 -*-
from enitites.tagged_token import TaggedToken


def get_tagged_tokens_from(dict_of_nes, token_list):
    """
    :param dict_of_nes: dict of objects
    :param token_list: list of tokens
    """
    list_of_tagged_tokens = [TaggedToken('O', token_list[i]) for i in range(len(token_list))]
    dict_of_tokens_with_indexes = {token_list[i].get_id(): i for i in range(len(token_list))}

    for ne in dict_of_nes.values():
        for tokenid in ne['tokens_list']:
            tag = format_tag(tokenid, ne)
            id_in_token_tuple = dict_of_tokens_with_indexes[tokenid]
            token = token_list[id_in_token_tuple]
            list_of_tagged_tokens[id_in_token_tuple] = TaggedToken(tag, token)
    # print(list_of_tagged_tokens)
    return list_of_tagged_tokens


def format_tag(tokenid, ne):
    bilou = __choose_bilou_tag_for(tokenid, ne['tokens_list'])
    formatted_tag = __tag_to_fact_ru_eval_format(ne['tag'])
    return bilou + formatted_tag


def __choose_bilou_tag_for(token_id, token_list):
    if len(token_list) == 1:
        return 'U'
    elif len(token_list) > 1:
        if token_list.index(token_id) == 0:
            return 'B'
        elif token_list.index(token_id) == len(token_list) - 1:
            return 'L'
        else:
            return 'I'


def __tag_to_fact_ru_eval_format(tag):
    if tag == 'Person':
        return 'PER'
    elif tag == 'Org':
        return 'ORG'
    elif tag == 'Location':
        return 'LOC'
    elif tag == 'LocOrg':
        return 'LOC'
    elif tag == 'Project':
        return 'ORG'
    else:
        raise ValueError('tag ' + tag + " is not the right tag")
