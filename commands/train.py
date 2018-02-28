import numpy as np

from gensim.models import Word2Vec

from keras.callbacks import TensorBoard
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from utils.logger import get_logger

import json
import os
from os import path


def process_data(dataset, dataset_label):
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
            sentence = list(map(lambda word: "{}/{}".format(*word), sentence))

            content = obj['content']
            content = list(map(lambda word: "{}/{}".format(*word), content))

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
    # Reading parsed news
    logger = get_logger()

    logger.info("[Fit] Reading parsed dataset...")
    news_list = os.listdir('./results/dataset/train')
    news_test_list = os.listdir('./results/dataset/test')

    [x_train, y_train] = process_data(news_list, "train")
    [x_test, y_test] = process_data(news_test_list, "test")
    logger.info("[Fit] Done reading %d train set & %d test sets!" % (len(x_train), len(x_test)))

    # Creating Word embedding model
    if not path.exists("./results/models/word2vec.txt"):
        logger.info("[Fit] Creating word2vec model...")

        w_model = Word2Vec(x_train, min_count=1, size=50, iter=10, sg=0)
        w_model.save("./results/models/word2vec.txt")

    else:
        logger.info("[Fit] Reading from saved word2vec model...")
        w_model = Word2Vec.load("./results/models/word2vec.txt")

    def bind_word(sentences):
        for sentence_index in range(0, len(sentences)):
            sentences[sentence_index] = list(filter(
                lambda word: word in w_model.wv.vocab,
                sentences[sentence_index]
            ))

            for word_index in range(0, len(sentences[sentence_index])):
                original_text = sentences[sentence_index][word_index]

                sentences[sentence_index][word_index] = w_model[original_text]

        return sentences

    # Preprocess input, outputs
    logger.info("[Fit] Preprocessing train dataset...")
    x_train = np.array(pad_sequences(bind_word(x_train), maxlen=20))
    y_train = list(map(np.array, y_train))

    logger.info("[Fit] Preprocessing test dataset...")
    x_test = np.array(pad_sequences(bind_word(x_test), maxlen=20))
    y_test = list(map(np.array, y_test))

    # Creating news model
    logger.info("[Fit] Generating model...")
    x = Input(shape=(20, 50))
    y = LSTM(20, activation='relu', name='lstm')(x)
    dense_ratio = Dense(10, activation='relu')(y)
    output_age = Dense(5, activation='softmax')(dense_ratio)
    output_gender = Dense(2, activation='softmax')(dense_ratio)

    dense_count = Dense(10, activation='relu')(y)
    output_count = Dense(2, activation='sigmoid')(dense_count)

    model = Model(inputs=[x], outputs=[output_age, output_gender, output_count])
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    model.fit(
        [x_train], y_train,
        validation_data=([x_test], y_test),
        batch_size=20, epochs=args.epoch, verbose=1,
        callbacks=[TensorBoard(log_dir='./results/logs/model')]
    )

    model.save("./results/models/news.h5")
