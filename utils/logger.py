import logging
import os

# from chalk.logging import ChalkHandler
from logging import FileHandler, Formatter, StreamHandler
from os import path


def get_logger_func():
    my_logger = None

    def get_logger_inner():
        nonlocal my_logger

        if my_logger:
            return my_logger

        if path.exists("./results/logs/deep-news.log"):
            os.remove("./results/logs/deep-news.log")

        file_formatter = Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
        file_handler = FileHandler('./results/logs/deep-news.log')
        file_handler.setFormatter(file_formatter)

        # stream_handler = ChalkHandler()
        stream_handler = StreamHandler()

        logger = logging.getLogger("DeepNews")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        my_logger = logger

        return logger

    return get_logger_inner

get_logger = get_logger_func()
