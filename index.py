import json
import os

from os import path
from parsers.list_parser import NewsListParser
from parsers.news_parser import NewsParser
from utils.arguments import get_arguments
from utils.logger import get_logger

if not path.isdir("./results"):
    os.mkdir("./results")


args = get_arguments()
logger = get_logger()

if args.list_crawl_again and path.exists("./results/news_list.json"):
    os.remove("./results/news_list.json")

if not path.exists("./results/news_list.json"):
    logger.info("Crawling news list.")

    list_parser = NewsListParser(logger)
    parsed_list = list_parser.parse_until(args.limit)

    f = open("./results/news_list.json", 'w')
    f.write(json.dumps(parsed_list))
    f.close()

else:
    logger.info("Using existing news list. Please add --list-crawl flag to re-crawl")

news_crawler = NewsParser(logger)

f = open("./results/news_list.json", 'r')
news_list = json.loads(f.read())
f.close()

if not path.exists("./results/news/"):
    os.mkdir("./results/news/")

for news in news_list:
    file_location = "./results/news/%s.json" % news

    if path.exists(file_location):
        logger.info("Skipping already crawled news: %s" % news)
        continue

    news_info = news_crawler.parse(news)

    if news_info is None:
        continue

    file = open(file_location, "w")
    file.write(json.dumps(news_info))
    file.close()

