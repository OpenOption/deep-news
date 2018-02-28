from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from utils.bind_word import bind_word
from utils.logger import get_logger
from utils.lstm_viewer import create_viewer

import numpy as np
import matplotlib.pyplot as plt
import os
import json


def run(args):
    news_list = os.listdir('./results/dataset/train')
    logger = get_logger()

    if len(news_list) < 1:
        logger.error("[Analyze] News should be crawled before!")
        return

    news = news_list[0]
    model = load_model('./results/models/news.h5')

    f = open('./results/dataset/train/%s' % news, 'r')
    obj = json.loads(f.read())
    f.close()

    word2vec = Word2Vec.load("./results/models/word2vec.txt")

    orig_sentence = obj['title']
    sentence = list(map(lambda word: "{}/{}".format(*word), orig_sentence + obj['content']))

    x_set = np.array(pad_sequences(bind_word([sentence], word2vec), maxlen=20))

    if args.check_word_diff:
        print(np.shape(x_set))

    if args.check_lstm:
        i = 0

        def get_callback(input_gate, forget_gate, output_gate):
            nonlocal i

            logger.debug("[Analyze::Check LSTM]", orig_sentence[i], input_gate, forget_gate, output_gate)
            i += 1

        lstm = model.get_layer("lstm")
        lstm.cell.call = create_viewer(get_callback)

        model.predict(x_set)
