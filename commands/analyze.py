from gensim.models import Word2Vec

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from konlpy.tag import Twitter

from matplotlib import pyplot as plt
from matplotlib import rcParams as rc_params

import numpy as np

from utils.bind_word import bind_word
from utils.logger import get_logger
from utils.lstm_viewer import create_viewer

import json
import os
import random


def get_color_grad(start, end, percentage):
    bg_percentage = 1 - percentage
    return list(map(lambda i: start[i] * percentage / 255 + end[i] * bg_percentage / 255, range(3)))


def run(args):
    news_list = os.listdir('./results/dataset/train')
    logger = get_logger()

    if len(news_list) < 1:
        logger.error("[Analyze] News should be crawled before!")
        return

    news = random.choice(news_list)
    model = load_model('./results/models/news.h5')

    f = open('./results/dataset/train/%s' % news, 'r')
    obj = json.loads(f.read())
    f.close()

    word2vec = Word2Vec.load("./results/models/word2vec.txt")

    if not args.checking_sentence:
        orig_sentence = obj['title']  # + obj['content']

    else:
        twitter = Twitter()
        orig_sentence = twitter.pos(args.checking_sentence, norm=True, stem=True)

    orig_sentence = list(filter(
        lambda x: x[1] not in ["Josa", "Eomi", "Punctuation", "KoreanParticle"], orig_sentence
    ))

    sentence = list(map(lambda word: "{}/{}".format(*word), orig_sentence))

    x_set = bind_word([sentence], word2vec)

    if args.check_word_diff:
        # Should not tested like this method.

        delta_set_x = []
        delta_set_y = []

        delta_set_colors = []

        previous_word_set = []
        previous_value = model.predict(np.zeros((1, 1000, 50)))[0][0][1]

        for index, word in enumerate(x_set[0]):
            previous_word_set.append(word)

            pad_word_set = np.array(pad_sequences([previous_word_set], maxlen=1000))
            new_value = model.predict(pad_word_set)[0][0][1]
            delta_value = new_value - previous_value
            previous_value = new_value

            delta_set_x.append(orig_sentence[index][0])
            delta_set_y.append(delta_value)

            if index % 100 == 0:
                logger.info("[Analyze::Word Diff] Progress %d/%d" % (index, len(x_set[0])))

        max_delta_value = max(delta_set_y)
        min_delta_value = min(delta_set_y)

        for delta_value in delta_set_y:
            if delta_value > 0:
                delta_set_colors.append(get_color_grad((197, 233, 155), (84, 134, 135), delta_value / max_delta_value))

            else:
                delta_set_colors.append(get_color_grad((239, 158, 159), (203, 117, 117), delta_value / min_delta_value))

        rc_params['font.family'] = ['KoPubDotum', 'NanumBarunGothic', 'Noto Sans CJK KR', 'Malgun Gothic']

        plt_width = 0.35
        plt_x_pos = np.arange(len(delta_set_x))

        plt.figure()
        plt.title('20대가 좋아하는 단어들')
        plt.bar(plt_x_pos, delta_set_y, plt_width, color=delta_set_colors, align='center')
        plt.xticks(plt_x_pos, delta_set_x)
        plt.show()

    if args.check_lstm:
        i = 0

        def get_callback(input_gate, forget_gate, output_gate):
            nonlocal i

            logger.debug("[Analyze::Check LSTM]", orig_sentence[i], input_gate, forget_gate, output_gate)
            i += 1

        lstm = model.get_layer("lstm")
        lstm.cell.call = create_viewer(get_callback)

        model.predict(pad_sequences(x_set, maxlen=1000))
