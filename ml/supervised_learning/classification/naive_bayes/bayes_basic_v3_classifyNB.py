'''
Source from: Peter,stack-overflow.
Modify,improve,integration: Jerrychen
'''

import bayes_basic_v1_before_log_p1p0 as bbv1
import bayes_basic_v2 as bbv2
import numpy as np
#vec2Classify is the vectory you want to classify it.
#other 3 args is get from trainNB0()
#
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    #due to log(a*b)=log(a)+log(b)
    #p(w0|ci)*p(w1|ci).... => log(p(w0|ci)) + log(p(w1|co)) + ....w1
    #that is why we use sum below

    #print(p1Vec)
    #print(vec2Classify)
    #print(vec2Classify* p1Vec)
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        #means belong category 1, abusive doc
        return 1
    else:
        return 0
    return 0



def testingNB():
    listOPosts,listClasses = bbv1.loadDataSet()
    myVocabList = bbv1.createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(bbv1.setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = bbv2.trainNB0(np.array(trainMat),np.array(listClasses))

    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(bbv1.setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(bbv1.setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

    return 0

if __name__ == "__main__":
    testingNB()

