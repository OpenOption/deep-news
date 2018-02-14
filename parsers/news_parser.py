import json
import re
import requests

from bs4 import BeautifulSoup


class NewsParser(object):
    news_link = "http://news.naver.com/main/read.nhn?oid=%s&aid=%s"
    jsonp_regex = re.compile(r'^\s*cb\s*\((.*)\)\s*;?\s*$', re.DOTALL)

    def __init__(self, logger):
        self.logger = logger

    def parse(self, news_id_token):
        self.logger.info('[Crawl::News Info] Parsing info of %s' % news_id_token)

        split = news_id_token.split(',')
        href = NewsParser.news_link % tuple(split)
        req = requests.get(href)

        soup = BeautifulSoup(req.text, 'html.parser')
        title_elem = soup.select_one('#articleTitle')

        if not title_elem:
            self.logger.info('[Crawl::News Info] %s has no title!' % news_id_token)
            return None

        title = title_elem.get_text()

        api_req = requests.get("https://apis.naver.com/commentBox/cbox/web_naver_list_jsonp.json", params={
            "_callback": "cb",
            "objectId": "news" + news_id_token,
            "pool": "cbox5",
            "ticket": "news",
            "lang": "ko",
            "initialize": "true",
            "pageSize": "1"  # Reduce packet size, pageSize will be ignored if it is less than one.
        }, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)" +
                          "Chrome/64.0.3282.167 Safari/537.36",

            "Referer": href  # Needed header
        })

        api_resp = json.loads(NewsParser.jsonp_regex.match(api_req.text).group(1))

        if ('result' not in api_resp) or\
           ('graph' not in api_resp['result']) or\
           ('gender' not in api_resp['result']['graph']) or\
           ('old' not in api_resp['result']['graph']):

            self.logger.info('[Crawl::News Info] %s has no graphs!' % news_id_token)
            return None

        gender_graph = api_resp['result']['graph']['gender']
        age_graph = api_resp['result']['graph']['old']
        age_parsed = {}

        for age in age_graph:
            age_parsed[age['age']] = age['value']

        return {
            'title': title,
            'age': age_parsed,
            'gender': gender_graph
        }
