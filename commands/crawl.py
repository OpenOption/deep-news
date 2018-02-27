import json
import os

from datetime import datetime
from os import path
from parsers.list_parser import NewsListParser
from parsers.news_parser import NewsParser
from utils.logger import get_logger


def run(args):
    logger = get_logger()

    news_list_path = "./results/dataset/news_list_%s.json" % args.target

    if args.list_crawl_again and path.exists(news_list_path):
        os.remove(news_list_path)

    if not path.exists(news_list_path):
        logger.info("[Crawl::News List] Crawling news list.")

        list_parser = NewsListParser(logger)

        if args.list_start:
            start_date = datetime.strptime(args.list_start, "%Y%m%d")
            parsed_list = list_parser.parse_until(args.limit, start_date)

        else:
            parsed_list = list_parser.parse_until(args.limit)

        f = open(news_list_path, 'w')
        f.write(json.dumps(parsed_list))
        f.close()

    else:
        logger.info("[Crawl] Using existing news list. Please add --force flag to re-crawl")

    news_crawler = NewsParser(logger)

    f = open(news_list_path, 'r')
    news_list = json.loads(f.read())
    f.close()

    for news in news_list:
        file_location = "./results/dataset/%s/%s.json" % (args.target, news)

        if path.exists(file_location):
            logger.info("[Crawl::News Info] Skipping already crawled news: %s" % news)
            continue

        news_info = news_crawler.parse(news)

        if news_info is None:
            continue

        file = open(file_location, "w")
        file.write(json.dumps(news_info))
        file.close()

