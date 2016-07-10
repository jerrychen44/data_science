'''
Source from: Peter,stack-overflow.
Modify,improve,integration: Jerrychen
'''

import bayes_topic_tendency_v1_getrss as brss
import bayes_topic_tendency_v1_2_webCrawler as bweb
import bayes_basic_v1_before_log_p1p0 as bbv1
import bayes_basic_v2 as bbv2
import bayes_basic_v3_classifyNB as bbv3
import pandas as pd
import numpy as np

def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    #print(sortedFreq)
    print(type(sortedFreq[:60]))

    #original code return here
    #return sortedFreq[:60]


    sortedFreq_filter_by_number=[]
    #add for filter the fequence by jerrychen
    freq_threshold=40
    for words in sortedFreq:
        if words[1] > freq_threshold:
            #print(words[1])
            sortedFreq_filter_by_number.append(words)

    my_select_list=[('here',1),('was',1),('other',1),('than',1),\
                    ('find',1),('into',1),('been',1),('how',1),\
                    ('when',1),('too',1),('any',1),('going',1)]
    sortedFreq_filter_by_number=sortedFreq_filter_by_number+my_select_list

    print(sortedFreq_filter_by_number)
    return sortedFreq_filter_by_number


def localWords(feed1,feed0):
    import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1),len(feed0))
    for i in range(minLen):
        #as usual , each row put into textParse
        wordList = textParse(feed1[i])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1

        wordList = textParse(feed0[i])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = bbv1.createVocabList(docList)#create vocabulary

    #remove top 30 words,you can change the number
    #that is import
    top30Words = calcMostFreq(vocabList,fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])

    #create test set
    trainingSet = list(range(2*minLen)); testSet=[]
    for i in range(20):
        randIndex = int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat=[]; trainClasses = []
    #train the classifier (get probs) trainNB0
    for docIndex in trainingSet:
        trainMat.append(bbv1.bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = bbv2.trainNB0(np.array(trainMat),np.array(trainClasses))

    errorCount = 0
    #classify the remaining items (testing set)
    for docIndex in testSet:
        wordVector = bbv1.bagOfWords2VecMN(vocabList, docList[docIndex])
        if bbv3.classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print ('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList,p1V,p0V


def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))

    #for print out
    keyword_number=20
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print ("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    show_number=0
    for item in sortedSF:
        print (item[0],item[1])
        show_number+=1
        if show_number ==keyword_number:
            break

    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print ("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    show_number=0
    for item in sortedNY:
        print (item[0],item[1])
        show_number+=1
        if show_number ==keyword_number:
            break


def main():

    #ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    #sf=feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    ny=bweb.web_crawler('http://newyork.craigslist.org/w4m','readfromlocal')
    sf=bweb.web_crawler('http://sfbay.craigslist.org/w4m','readfromlocal')
    #print(ny)
    '''
    #vocabList,pNY,pSF=localWords(ny,sf)
    #print(len(vocabList),pNY,pSF)

    #vocabList,pSF,pNY=localWords(ny,sf)
    #print(vocabList,pSF,pNY)
    '''
    getTopWords(ny,sf)

    return 0


if __name__ == "__main__":
    main()



