import argparse
import re


def date_type(s):
    date_pattern = re.compile(r"^([12][0-9]{3}(?:[01][0-9])(?:[0123][0-9]))$")
    if not date_pattern.match(s):
        raise argparse.ArgumentTypeError

    return s


def get_arguments():
    arg_parser = argparse.ArgumentParser(description='Naver news crawler')
    arg_parser.add_argument(
        'limit', type=int, help='Crawl until the number of collected data exceeds this limit',
        default=300, nargs='?'
    )

    arg_parser.add_argument(
        '--list-crawl', help='Crawl news list regardless of whether the list is crawled before',
        dest='list_crawl_again', action='store_true'
    )

    arg_parser.add_argument(
        "--list-start", help='Start crawling news list from this date',
        dest='list_start', type=date_type
    )

    args = arg_parser.parse_args()

    return args
