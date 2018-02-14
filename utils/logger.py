import logging
import os

# from chalk.logging import ChalkHandler
from logging import FileHandler, Formatter, StreamHandler
from os import path


def get_logger():
    if path.exists("./results/deepnews.log"):
        os.remove("./results/deepnews.log")

    file_formatter = Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
    file_handler = FileHandler('./results/deepnews.log')
    file_handler.setFormatter(file_formatter)

    # stream_handler = ChalkHandler()
    stream_handler = StreamHandler()

    logger = logging.getLogger("DeepNews")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
