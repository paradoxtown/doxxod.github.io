---
layout:     post
title:      scrapy学习
subtitle:   学习自大邓
date:       2018-02-17
author:     paradox
header-img: img/post_1.jpg
catalog: true
mathjax: false
tags:
    - spider
    - scrapy
---

### 项目创建

`scrapy`新建项目:`scrapy startproject [project name] [path]`
会生成以下文件：

### `livespider`项目解读

| 文件或文件夹    | 功能                                                         |
| --------------- | ------------------------------------------------------------ |
| `spiders`文件夹 | 是用来存放爬虫逻辑的文件夹                                   |
| `items.py`      | spider文件夹中的爬虫脚本解析数据，通过`items.py`组织数据形式 |
| `pipeline.py`   | 爬虫解析出的数据，通过`items.py`封装后传递给管道，留到数据库或数据文件。`pipeline`实际上就是配置数据库或数据文件。 |
| `setting.py`    | 整个爬虫项目的配置文件                                       |
| `middleware.py` | 一般不怎么用                                                 |

### 创建爬虫
在`spiders`文件夹里创建爬虫`spider`：
`scrapy genspider [spider name] [url]`

### `setting.py`设置
对项目进行全局配置。([简书-setting.py设置](https://www.jianshu.com/p/df9c0d1e9087))

### `items.py`配置
`items.py`的功能相当于怕中解析出数据需要一个结构化的容器封装，方便传递给pipelines使用。

### `pipelines.py`配置
按照以下步骤
- 爬虫启动时，要打开数据库
- 爬虫运行时，要将数据写入数据库
- 爬虫结束的时候要关闭数据库

```python
def open_spider(self, spider):
    self.con = sqlite3.connect('绝对路径')
    self.con = sqlite3.con.cursor()
    #初始化爬虫，开启数据库，或者创建一个csv文件
    
def process_item(self, item, spider):
    #写入数据库或数据文件
    title = item['title']
    speaker = item['speaker']
    sql_command = "INSERT INTO LiveTable (title,speaker) VALUES ('{title}','{speaker}')".format(title=title, speaker=speaker)
    self.cur.execute(sql_command)
    self.con.commit()
    return item

def close_spider(self, spider):
    #关闭数据库或者数据文件
    self.con.close()
```

### `spider`的书写
`scrapy`框架有默认的请求函数`start_request`，该函数对网站发起请求，`yeild`方式返回的相应数据传递给`spider`中的`parse`函数解析。
```python
def start_requests(self):
    starturl = 'https://api.zhihu.com/lives/homefeed?limit=10&offset=10&includes=live'
    yield Request(url=starturl, callback=self.parse)
```
`parse`函数负责解析数据，将数据以`item`形式封装，并以`yield`方式传输给`pipelines`，最终写入数据库。
`yield`其实很像`return`，只不过，`return`后会结束函数。而`yield`的函数相当于生成器，返回结果后并不直接结束函数，而是保留状态，下次运行函数会按照上次的位置继续迭代下去.
```python
from scrapy import Spider,Request
from livespider.items import LivespiderItem
import json

class ZhihuSpider(Spider):
    name = 'zhihu'
    allowed_domains = ['zhihu.com']
    start_urls = ['http://zhihu.com/']

    def start_requests(self):
        starturl = 'https://api.zhihu.com/lives/homefeed?limit=10&offset=10&includes=live'
        yield Request(url=starturl, callback=self.parse)
        
    def parse(self, response):
        item = LivespiderItem()
        result = json.loads(response.text)
        records = result['data']
        for record in records:
            item['title'] = record['live']['subject']
            item['speaker'] = record['live']['speaker']['member']['name']
            
            #将item传给pipelines.py保存到数据库
            yield item
        
        next_page_url = result['paging']['next']+'&includes=live'
        
        #如果网址里有50,就停止函数。
        if '50' in next_page_url:
            return True
            
        #parse自身的递归
        yield Request(url=next_page_url, callback=self.parse)
```

最后用终端运行
`scrapy crawl zhihu`


