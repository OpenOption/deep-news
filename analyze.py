from keras.models import load_model
from gensim.models import Word2Vec
from utils.lstm_viewer import create_viewer

import os
import json


def analyze():
    news = os.listdir('./results/news/')[0]

    f = open('./results/news/%s' % news, 'r')
    obj = json.loads(f.read())
    f.close()

    word2vec = Word2Vec.load("./results/models/word2vec.txt")

    orig_sentence = obj['title']
    sentence = list(map(lambda word: word2vec["{}/{}".format(*word)], orig_sentence))

    i = 0

    def get_callback(input_gate, forget_gate, output_gate):
        nonlocal i
        print(orig_sentence[i], input_gate, forget_gate, output_gate)

        i += 1

    model = load_model('my_model.h5')
    lstm = model.get_layer("lstm")
    lstm.cell.call = create_viewer(get_callback)
    model.predict(sentence)
