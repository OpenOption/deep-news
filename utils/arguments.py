import argparse


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

    args = arg_parser.parse_args()

    return args
