
from functools import reduce

from gensim.models import Word2Vec

from keras.callbacks import TensorBoard
from keras.layers import Input, LSTM, Dense, Dropout
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences

from os import path

from utils.bind_word import bind_word
from utils.logger import get_logger

import json
import os
import numpy as np


def filter_map_word(no_particles):
    def applicator(prev, curr):
        if no_particles and curr[1] in ["Josa", "Eomi", "Punctuation", "KoreanParticle"]:
            return prev

        prev.append("{}/{}".format(*curr))
        return prev

    return applicator


def process_data(args, dataset, dataset_label):
    x_set = []
    y_set = [[], [], []]
    logger = get_logger()

    for news in dataset:
        try:
            f = open('./results/dataset/%s/%s' % (dataset_label, news), 'r')
            obj = json.loads(f.read())
            f.close()

            age = obj['age']
            gender = obj['gender']

            sentence = obj['title']
            sentence = list(reduce(filter_map_word(args.no_particles), sentence, []))

            content = obj['content']
            content = list(reduce(filter_map_word(args.no_particles), content, []))

            x_set.append(sentence + content)
            y_set[0].append([
                float(age['10']) / 100,
                float(age['20']) / 100,
                float(age['30']) / 100,
                float(age['40']) / 100,
                float(age['50']) / 100
            ])

            y_set[1].append([
                float(gender['male']) / 100,
                float(gender['female']) / 100
            ])

            y_set[2].append([
                float(obj['comment']) / 50000,
                float(sum(obj['reaction'].values())) / 50000
            ])

        except IOError:
            logger.error("[Fit] Error while reading dataset %s!" % news)

        except KeyError:
            logger.error("[Fit] Error on dataset %s!" % news)

    return x_set, y_set


def run(args):
    logger = get_logger()

    seq_size = args.seq_size
    batch_size = args.batch_size

    logger.info("[Fit] Using sequence size %d, batch size %d" % (seq_size, batch_size))

    # Creating news model
    logger.info("[Fit] Generating model...")

    x = Input(shape=(seq_size, 50))
    lstm_layer = LSTM(20, activation='relu', name='lstm', dropout=0.2)
    lstm = lstm_layer(x)

    shared_model_output = lstm_layer.output_shape

    age_model = Sequential([
        Dense(10, activation='relu', name='dense_age', input_shape=shared_model_output),
        Dropout(0.3, name='dropout_age'),
        Dense(5, activation='softmax', name='output_age')
    ])(lstm)

    gender_model = Sequential([
        Dense(10, activation='relu', name='dense_gender', input_shape=shared_model_output),
        Dropout(0.3, name='dropout_gender'),
        Dense(2, activation='softmax', name='output_gender')
    ])(lstm)

    count_model = Sequential([
        Dense(10, activation='relu', name='dense_count', input_shape=shared_model_output),
        Dense(10, activation='relu', name='dense_count_2'),
        Dropout(0.3, name='dropout_count'),
        Dense(2, activation='sigmoid', name='output_count')
    ])(lstm)

    model = Model(inputs=[x], outputs=[age_model, gender_model, count_model])
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Reading parsed news

    logger.info("[Fit] Reading parsed dataset...")
    news_list = os.listdir('./results/dataset/train')
    news_test_list = os.listdir('./results/dataset/test')

    [x_train, y_train] = process_data(args, news_list, "train")
    [x_test, y_test] = process_data(args, news_test_list, "test")
    logger.info("[Fit] Done reading %d train set & %d test sets!" % (len(x_train), len(x_test)))

    # Creating Word embedding model
    if not path.exists("./results/models/word2vec.txt"):
        logger.info("[Fit] Creating word2vec model...")

        w_model = Word2Vec(x_train, min_count=1, size=50, iter=10, sg=0)
        w_model.save("./results/models/word2vec.txt")

    else:
        logger.info("[Fit] Reading from saved word2vec model...")
        w_model = Word2Vec.load("./results/models/word2vec.txt")

    # Preprocess input, outputs
    logger.info("[Fit] Preprocessing train dataset...")
    x_train = np.array(pad_sequences(bind_word(x_train, w_model), maxlen=seq_size))
    y_train = list(map(np.array, y_train))

    logger.info("[Fit] Preprocessing test dataset...")
    x_test = np.array(pad_sequences(bind_word(x_test, w_model), maxlen=seq_size))
    y_test = list(map(np.array, y_test))

    # Fit the model
    logger.info("[Fit] Fitting the model...")
    model.fit(
        [x_train], y_train,
        validation_data=([x_test], y_test),
        batch_size=batch_size, epochs=args.epoch, verbose=1,
        callbacks=[TensorBoard(log_dir='./results/logs/model')]
    )

    model.save("./results/models/news.h5")
