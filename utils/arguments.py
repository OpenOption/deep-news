import argparse
import re


def date_type(s):
    date_pattern = re.compile(r"^([12][0-9]{3}(?:[01][0-9])(?:[0123][0-9]))$")
    if not date_pattern.match(s):
        raise argparse.ArgumentTypeError

    return s


def dataset_type(s):
    if not(s == 'train' or s == 'test'):
        raise argparse.ArgumentTypeError

    return s


def get_arguments():
    arg_parser = argparse.ArgumentParser(
        description='A LSTM model to predict news-related information by naver news dataset.'
    )
    sub_parser = arg_parser.add_subparsers(help='Commands')

    # Crawling sub commands
    crawl_parser = sub_parser.add_parser('crawl', help='Crawl dataset from naver news.')
    crawl_parser.set_defaults(which='crawl')

    crawl_parser.add_argument(
        'limit', type=int, help='Crawl until the number of collected data exceeds this limit',
        default=300, nargs='?'
    )

    crawl_parser.add_argument(
        'target', type=dataset_type, help='Crawled results will be saved at target folder. Should be train or test.',
        default='train', nargs='?'
    )

    crawl_parser.add_argument(
        "--start", type=date_type, help='Start crawling news list from this date',
        dest='list_start', metavar='date'
    )

    crawl_parser.add_argument(
        '--force', help='Crawl news list regardless of whether the list is crawled before',
        dest='list_crawl_again', action='store_true'
    )

    # Training sub commands
    train_parser = sub_parser.add_parser('fit', help='Train from crawled dataset.')
    train_parser.set_defaults(which='train')
    train_parser.add_argument(
        "epoch", type=int, help='Epochs used in training',
        default=100, nargs='?'
    )

    # Analyzing sub commands
    analyze_parser = sub_parser.add_parser('analyze', help='A cutting edge command to check if the model works.')
    analyze_parser.set_defaults(which='analyze')
    analyze_parser.add_argument(
        "--check-lstm", help='Check value of LSTM Gates.',
        dest='check_lstm', action='store_true'
    )

    args = arg_parser.parse_args()

    return args
