import feedparser
import pandas as pd


def get_rss(rss_url):
    #rss can only get fix artical ex:25 once,it depend on web-provider.
    #I try to use web cwaler instead.

    #rss_url='http://newyork.craigslist.org/stp/index.rss'
    #rss_url='http://feedparser.org/docs/examples/atom10.xml'
    print(rss_url)
    rss=feedparser.parse(rss_url)

    #print(rss['entries'][0]['published'])

    #get the summary field
    #print((rss['entries'][1]['summary']))
    return rss


if __name__ == "__main__":

    rss_url='http://newyork.craigslist.org/stp/index.rss'
    #rss_url='http://sfbay.craigslist.org/stp/index.rss'

    #rss_url='https://newyork.craigslist.org/search/w4m?format=rss'
    rss=get_rss(rss_url)
    print(rss.keys())
    #print(rss['entries'])
    print(len(rss['entries']))
    #print(rss['entries'][1]['title_detail']['value'],(rss['entries'][1]['published']))
    print((type(rss['entries'][1]['summary'])))


    #content=pd.DataFrame(rss['entries'])
    #print(content)
    #for item in rss['entries']:
    #    print((item))
    #    print('================================')



