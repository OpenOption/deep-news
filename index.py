import importlib
import os
from os import path

from utils.arguments import get_arguments


def check_and_create_dir(dir_name):
    if not path.isdir(dir_name):
        os.mkdir(dir_name)

args = get_arguments()
check_and_create_dir("./results/")
check_and_create_dir("./results/news/")
check_and_create_dir("./results/news/train/")
check_and_create_dir("./results/news/test/")
check_and_create_dir("./results/logs/")
check_and_create_dir("./results/logs/model/")
check_and_create_dir("./results/models/")

command = importlib.import_module("commands.%s" % args.which)
command.run()
