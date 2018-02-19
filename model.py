import numpy as np

from gensim.models import Word2Vec

import tensorflow as tf

from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Model

from keras.preprocessing.sequence import pad_sequences

import json
import os


sentences = []
y_train = []
y_train2 = []

news_list = os.listdir('./results/news/')

for news in news_list:
    try:
        f = open('./results/news/%s' % news, 'r')
        obj = json.loads(f.read())
        f.close()

        age = obj['age']
        gender = obj['gender']

        sentence = obj['title']
        sentence = filter(lambda word: not word[1] in ["Josa", "Eomi", "Punctuation", "KoreanParticle"], sentence)
        sentence = list(map(lambda word: word[0], sentence))

        sentences.append(sentence)
        y_train.append([float(age['10']) / 100, float(age['20']) / 100, float(age['30']) / 100, float(age['40']) / 100, float(age['50']) / 100])
        y_train2.append([float(gender['male']) / 100, float(gender['female']) / 100])
    except IOError:
        pass


w_model = Word2Vec(sentences, min_count=1, size=50, iter=10, sg=0)

for i in range(0, len(sentences)):
    for n in range(0, len(sentences[i])):
        sentences[i][n] = w_model[sentences[i][n]]



x_train = pad_sequences(sentences, maxlen=20)
x_train = np.array(x_train)

y_train = np.array(y_train)
y_train2 = np.array(y_train2)


input = Input(shape=(20, 50))
lstm = LSTM(20, activation='relu')(input)
output = Dense(5, activation='softmax')(lstm)
output2 = Dense(2, activation='softmax')(lstm)

model = Model(input, output)
model.compile(loss='categorical_crossentropy', optimizer='adagrad')

model2 = Model(input, output2)
model2.compile(loss='binary_crossentropy', optimizer='adagrad')


for i in range(0, 10):
    model.fit(x_train, y_train, batch_size=20, epochs=1, verbose=1)
    model2.fit(x_train, y_train2, batch_size=20, epochs=1, verbose=1)
