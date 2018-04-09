# -*- coding: utf-8 -*-


class Token:
    def __init__(self, tokenid, position, length, text):
        self.__id = tokenid
        self.__position = position
        self.__length = length
        self.__text = text

    def get_all(self):
        return self.__id, self.__position, self.__length, self.__text

    def get_id(self):
        return self.__id

    def get_position(self):
        return self.__position

    def get_length(self):
        return self.__length

    def get_text(self):
        return self.__text

    def __str__(self):
        return self.__text

    def __repr__(self):
        return "<<" + self.__id + "_" + self.__text + ">>"
