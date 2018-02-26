from keras.models import load_model
from gensim.models import Word2Vec
from utils.logger import get_logger
from utils.lstm_viewer import create_viewer

import os
import json


def run(args):
    news_list = os.listdir('./results/news/train')
    logger = get_logger()

    if len(news_list) < 1:
        logger.error("[Analyze] News should be crawled before!")
        return

    news = news_list[0]

    if args.check_lstm:
        f = open('./results/news/train/%s' % news, 'r')
        obj = json.loads(f.read())
        f.close()

        word2vec = Word2Vec.load("./results/models/word2vec.txt")

        orig_sentence = obj['title']
        sentence = list(map(lambda word: word2vec["{}/{}".format(*word)], orig_sentence))

        i = 0

        def get_callback(input_gate, forget_gate, output_gate):
            nonlocal i

            logger.debug("[Analyze::Check LSTM]", orig_sentence[i], input_gate, forget_gate, output_gate)
            i += 1

        model = load_model('my_model.h5')
        lstm = model.get_layer("lstm")
        lstm.cell.call = create_viewer(get_callback)
        model.predict(sentence)
