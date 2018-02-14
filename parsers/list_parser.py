import re
import requests
from datetime import date, timedelta
from bs4 import BeautifulSoup


class NewsListParser(object):
    oid_regex = re.compile(r"oid=(\d+)")
    aid_regex = re.compile(r"aid=(\d+)")

    def __init__(self, logger):
        self.logger = logger

    def parse(self, parse_date):
        req = requests.get(
            'http://news.naver.com/main/ranking/popularDay.nhn',
            params={'rankingType': 'popular_day', 'date': parse_date}
        )

        soup = BeautifulSoup(req.text, 'html.parser')
        elements = soup.select('.ranking_section ol li dt a')

        self.logger.debug("[Crawl::News List] Scraped %d news at %s" % (len(elements), parse_date))

        return list(map(self.parse_url, elements))

    def parse_until(self, limit, start_date=date.today()):
        parsed_list = []
        minus = 0

        while len(parsed_list) < limit:
            delta = timedelta(days=minus)

            parse_date = (start_date - delta).strftime("%Y%m%d")
            for data in self.parse(parse_date):
                if data is None:
                    continue

                if data not in parsed_list:
                    parsed_list.append(data)

            minus += 1

        return parsed_list

    @staticmethod
    def parse_url(tag):
        href = tag['href']
        oid = NewsListParser.oid_regex.search(href)
        aid = NewsListParser.aid_regex.search(href)

        if not oid or not aid:
            return None

        return "%s,%s" % (oid.group(1), aid.group(1))
