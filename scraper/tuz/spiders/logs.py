# -*- coding: utf-8 -*-
import datetime as dt
import json

import pandas as pd
import scrapy

TOP_HEADERS = {
    'user-agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/53.0.2785.143 Safari/537.36'
    )
}


class LogsSpider(scrapy.Spider):
    name = 'logs'
    allowed_domains = ['tuzach.in']

    def start_requests(self):
        for date in pd.date_range(dt.date(2012, 1, 29), dt.date.today()):
            url = 'https://tuzach.in/api/?app=logs&log={}'.format(date.strftime('%Y-%-m-%-d'))
            yield scrapy.Request(url=url, headers=TOP_HEADERS, callback=self.parse)

    def parse(self, response):
        yield {'data': json.loads(response.body)}
