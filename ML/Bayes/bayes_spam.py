# -*- coding: utf-8 -*-
'''
Source from: Peter,stack-overflow.
Modify,improve,integration: Jerrychen
'''

import os,sys
#print(sys.getdefaultencoding())
filepath=os.path.dirname(os.path.realpath(__file__))#root
import bayes_basic_v1_before_log_p1p0 as bbv1
import bayes_basic_v2 as bbv2
import bayes_basic_v3_classifyNB as bbv3
import numpy as np

def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]



def spamTest():

    docList=[]; classList = []; fullText =[]
    #get all email txt in.
    #wordList is the current scaned email content, with important wording.important
    #docList just collect all email content by list, each element is a content of email.
    #fullText just collect all word from all emails in a big list. so it use extend.
    #classList is a list indicat that email is spam or not spam.
    for i in range(1,26):
        path=filepath+'/email/spam/%d.txt' % i
        temp=open(path).read()
        #print(type(temp),'spam/'+str(i)+'.txt')
        wordList = textParse(temp)
        #print(wordList)
        docList.append(wordList)
        #print(docList)
        fullText.extend(wordList)
        #print(fullText)
        classList.append(1)
        #print(classList)


        path=filepath+'/email/ham/%d.txt' % i
        temp=open(path).read()


        #print(type(temp),'ham/'+str(i)+'.txt')
        wordList = textParse(temp)
        #print(wordList)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    total_email_number=len(docList)
    #print(len(docList))
    #print(docList)
    #print(fullText)
    #print(classList)


    #create vocabulary
    # let all input data words to become an word of set.
    vocabList_set = bbv1.createVocabList(docList)
    #print(vocabList_set)
    #trainingSet = range(50);#for python 2.7
    trainingSet = list(range(total_email_number));
    testSet=[]           #create test set
    #print(trainingSet)

    #randomize to create the test set
    # pick 10 random number for 1~50 in to testSet.
    #ex:testSet=[25, 38, 15, 1, 22, 6, 31, 3, 48, 13]
    # and you will get trainginSet at the same time, due to it delete item which selected out
    # to testing. So trainingSet will have the rest of the number.
    # we pick 10 email to do the testing later. which means we will train 50-10=40 email.
    # we just need the index, we can find the original artical by docList[index]
    number_of_test_email=10
    for i in range(number_of_test_email):
        randIndex = int(np.random.uniform(0,len(trainingSet)))
        #print("randIndex=%d" % randIndex)
        testSet.append(trainingSet[randIndex])
        #print(testSet)
        del(trainingSet[randIndex])
        #print(trainingSet,len(trainingSet))

    trainMat=[]; trainClasses = []

    #train the classifier (get probs) trainNB0
    for docIndex in trainingSet:
        trainMat.append(bbv1.bagOfWords2VecMN(vocabList_set, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V,p1V,pSpam = bbv2.trainNB0(np.array(trainMat),np.array(trainClasses))



    errorCount = 0

    #classify the remaining items
    for docIndex in testSet:
        wordVector = bbv1.bagOfWords2VecMN(vocabList_set, docList[docIndex])
        #print(wordVector)
        classify_result=bbv3.classifyNB(np.array(wordVector),p0V,p1V,pSpam)
        if  classify_result != classList[docIndex]:
            errorCount += 1
            print("email=",docList[docIndex])
            print ("classification error, should be %d -> predict to %d" % (classList[docIndex],classify_result))

    print("errorCount=",float(errorCount))
    print("testSet=",len(testSet))
    print ('the error rate is: ',float(errorCount)/len(testSet))

    return 0


if __name__ == "__main__":
    spamTest()