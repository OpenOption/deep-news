import numpy as np

from gensim.models import Word2Vec

from keras.callbacks import TensorBoard
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

import json
import os
from os import path


def process_data(dataset_label, dataset):
    age_set = []
    gender_set = []
    sentence_set = []

    for news in dataset:
        try:
            f = open('./results/dataset/%s/%s' % (dataset_label, news), 'r')
            obj = json.loads(f.read())
            f.close()

            age = obj['age']
            gender = obj['gender']

            sentence = obj['title']
            sentence = list(map(lambda word: "{}/{}".format(*word), sentence))

            sentence_set.append(sentence)
            age_set.append([
                float(age['10']) / 100,
                float(age['20']) / 100,
                float(age['30']) / 100,
                float(age['40']) / 100,
                float(age['50']) / 100
            ])

            gender_set.append([
                float(gender['male']) / 100,
                float(gender['female']) / 100
            ])
        except IOError:
            pass

    return age_set, gender_set, sentence_set


def run(args):
    # Reading parsed news
    news_list = os.listdir('./results/dataset/train')
    news_test_list = os.listdir('./results/dataset/test')

    [y_train, y_train2, x_train] = process_data(news_list, "train")
    [y_test, y_test2, x_test] = process_data(news_test_list, "test")

    # Creating Word embedding model
    if not path.exists("./results/models/word2vec.txt"):
        w_model = Word2Vec(x_train, min_count=1, size=50, iter=10, sg=0)
        w_model.save("./results/models/word2vec.txt")

    else:
        w_model = Word2Vec.load("./results/models/word2vec.txt")

    def bind_word(sentences):
        for sentence_index in range(0, len(sentences)):
            for word_index in range(0, len(sentences[sentence_index])):
                original_text = sentences[sentence_index][word_index]

                sentences[sentence_index][word_index] = w_model[original_text]

            sentences[sentence_index] = list(filter(
                lambda word: word in w_model.vocabulary,
                sentences[sentence_index]
            ))

        return sentences

    # Preprocess input, outputs
    x_train = np.array(pad_sequences(bind_word(x_train), maxlen=20))
    y_train = np.array(y_train)
    y_train2 = np.array(y_train2)

    x_test = np.array(pad_sequences(bind_word(x_test), maxlen=20))
    y_test = np.array(y_test)
    y_test2 = np.array(y_test2)

    # Creating news model
    x = Input(shape=(20, 50))
    y = LSTM(20, activation='relu', name='lstm')(x)
    output = Dense(5, activation='softmax')(y)
    output2 = Dense(2, activation='softmax')(y)

    model = Model(inputs=[x], outputs=[output, output2])
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    model.fit(
        [x_train], [y_train, y_train2], validation_data=([x_test], [y_test, y_test2]),
        batch_size=20, epochs=args.epoch, verbose=1,
        callbacks=[TensorBoard(log_dir='./results/logs/model')]
    )

    model.save("./results/models/news.h5")
