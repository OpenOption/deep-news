import os
from os import path

from model import train
from parse import parse


def check_and_create_dir(dir_name):
    if not path.isdir(dir_name):
        os.mkdir(dir_name)


check_and_create_dir("./results/")
check_and_create_dir("./results/news/")
check_and_create_dir("./results/logs/")
check_and_create_dir("./results/logs/model/")
check_and_create_dir("./results/models/")

parse()
train()
