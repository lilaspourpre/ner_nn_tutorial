# -*- coding: utf-8 -*-
from enitites.models.rnn_model import RNNModel
from machine_learning.i_model_trainer import ModelTrainer
from wrapper import complement_data, format_data
from tqdm import tqdm
import tensorflow as tf


class RNNTrainer(ModelTrainer):
    def __init__(self, epoch, nn, tags):
        super().__init__()
        self.__epoch = epoch
        self.__nn = nn
        self.__tags = tags

    def batch_train(self, tagged_vectors, division):
        vectors = [tagged_vector.get_vector() for tagged_vector in tagged_vectors]
        splitted_vectors, seqlen_list = format_data(vectors, division)
        splitted_tags, _ = format_data(
            self.translator([tagged_vector.get_tag() for tagged_vector in tagged_vectors], self.__tags), division)
        return self.run_nn(splitted_vectors, splitted_tags, seqlen_list)

    def run_nn(self, array_of_vectors, array_of_tags, seqlen_list):
        for m in range(self.__epoch):
            k = int(len(array_of_vectors) / self.__nn.batch_size) if len(array_of_vectors) % self.__nn.batch_size == 0 \
                else int(len(array_of_vectors) / self.__nn.batch_size) + 1
            step = 0
            loss = 0
            pbar = tqdm(range(k))
            for j in pbar:
                pbar.set_description("loss: "+str(loss)+", batch: "+str(j)+", epoch: "+str(m))
                array_x = complement_data(array_of_vectors[step:step + self.__nn.batch_size])
                array_y = complement_data(array_of_tags[step:step + self.__nn.batch_size])
                loss, _ = self.__nn.sess.run([self.__nn.cross_entropy, self.__nn.train],
                                             {self.__nn.x: array_x,
                                              self.__nn.y: array_y,
                                              self.__nn.seqlen: seqlen_list[step:step + self.__nn.batch_size]
                                              })
                step += self.__nn.batch_size

            print("\nloss: %s epoch: %s " % (loss, m))

        return RNNModel(self.__nn.sess, self.__nn.outputs, self.__nn.x, self.__nn.seqlen, self.__tags)
