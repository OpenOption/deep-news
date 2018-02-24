import numpy as np

from gensim.models import Word2Vec

from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

import json
import os


# Reading parsed news
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
        y_train.append([
            float(age['10']) / 100,
            float(age['20']) / 100,
            float(age['30']) / 100,
            float(age['40']) / 100,
            float(age['50']) / 100
        ])

        y_train2.append([
            float(gender['male']) / 100,
            float(gender['female']) / 100
        ])
    except IOError:
        pass


# Creating Word embedding model
w_model = Word2Vec(sentences, min_count=1, size=50, iter=10, sg=0)

for i in range(0, len(sentences)):
    for n in range(0, len(sentences[i])):
        sentences[i][n] = w_model[sentences[i][n]]


# Creating news model
x_train = pad_sequences(sentences, maxlen=20)
x_train = np.array(x_train)

y_train = np.array(y_train)
y_train2 = np.array(y_train2)


x = Input(shape=(20, 50))
y = LSTM(20, activation='relu')(x)
output = Dense(5, activation='softmax')(y)
output2 = Dense(2, activation='softmax')(y)

model = Model(input=x, output=[output, output2])
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(x_train, [y_train, y_train2], batch_size=20, epochs=10, verbose=1)
