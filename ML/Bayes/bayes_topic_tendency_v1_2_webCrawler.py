import feedparser
import requests,os
import pandas as pd
from bs4 import BeautifulSoup
filepath=os.path.dirname(os.path.realpath(__file__))#root



def get_url_content_list(url):

    res = requests.get(url)
    #print (res.text.encode('utf-8'))
    #print(res.text.decode)
    soup = BeautifulSoup(res.text, 'html.parser')
    #print(soup.text.encode('utf-8'))
    #print(url)
    content_str=soup.select('#postingbody')[0].text
    #print(content_str)
    return content_str.strip()

def web_crawler(url,mode):

    if mode=='online2local':
        res = requests.get(url)
        #print (res.text.encode('utf-8'))
        #print(res.text.decode)
        soup = BeautifulSoup(res.text, 'html.parser')
        #print(soup.text.encode('utf-8'))

        #print(soup.select('.rows a'))
        linklist=[]
        for link in soup.select('.rows .pl a'):

            newurl=url.rsplit('/', 1)[0]
            linklist.append(newurl+link['href'])

        #print(linklist)

        #print(len(linklist))
        #for item in soup.select('.rows'):
        #    print(item)
        artical_list=[]
        #key=0
        for links in linklist:
            artical_list.append(get_url_content_list(links))
            #key+=1
            #if key ==5:
            #    break

        print(len(artical_list))
        artical_df=pd.DataFrame(artical_list)
        artical_df.to_csv(filepath+'/data/'+url.split('.')[0].split('/')[-1]+'.csv')
    else:#read from local

        artical_df=pd.read_csv(filepath+'/data/'+url.split('.')[0].split('/')[-1]+'.csv',index_col=0,header=0)
        artical_list=artical_df['0'].tolist()
        #print(artical_list[0])

    return artical_list



if __name__ == "__main__":
    #url='http://newyork.craigslist.org/w4m'
    url='http://sfbay.craigslist.org/w4m'
    #newurl=url.rsplit('/', 1)
    #print(newurl[0])

    web_crawler(url,'readfromlocal')#online2local or readfromlocal

    #get_url_content_list('http://newyork.craigslist.org/jsy/w4m/5674201346.html')
    #print(url.split('.')[0].split('/')[-1])
    #print(filepath+'/'+url.split('.')[0].split('/')[-1]+'.csv')



