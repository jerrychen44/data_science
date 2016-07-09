'''
Source from: Peter,stack-overflow.
Modify,improve,integration: Jerrychen
'''

import numpy as np

def loadDataSet():
    #row is each data, column is the selected words.
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please', 'dog'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

# let all input data words to become an word of bag.
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set,un-repeted words bag
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    #vocabSet is a whole word of bag for all loadDataSet
    #return as a list ,because set has no order.
    return list(vocabSet)
#give a data m (with some of words), will mark the the word to 1 if appear in word of bag.
def setOfWords2Vec(vocabList, inputSet):
    #returnVec cotent is all 0
    returnVec = [0]*len(vocabList)
    #print(inputSet)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print ("the word: %s is not in my Vocabulary!" % word)
    return returnVec

#will count the frequence of the word, ex: 'dog' appear in artical twice, the vector
#will be 2.
#that will use as weight when you calculate the p(). more appeared, more effect
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        #else: print ("the word: %s is not in my Vocabulary!" % word)
    #print(returnVec)
    return returnVec

def trainNB0_original(trainMatrix,trainCategory):
    #initial probability
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    print(numTrainDocs,numWords)
    #sum(trainCategory) is to count how many of 1.
    #pAbusive = (total abusive doc /total number of data) => p(abusive) =>p(category_i)
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    print(sum(trainCategory),pAbusive)
    p0Num = np.zeros(numWords)      #
    p1Num = np.zeros(numWords)      #change to ones()
    #print(p0Num)

    #pXDenom is the total count of word appears number.
    #p0Denom is for category 0, not abusive doc, after go through them, total appears count.0
    #p1Denom is whole appeared words count in category 1.
    p0Denom = 0.0
    p1Denom = 0.0                        #change to 2.0

    print(trainCategory)
    for i in range(numTrainDocs):
        #==1 means this is abusive category
        if trainCategory[i] == 1:
            #vector plus
            #print("111111111111111111")
            #print(p1Num,trainMatrix[i])
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
            #print(p1Num)
            #print(p1Denom)
        else:
            #print("0000000000000000000")
            #print(p0Num,trainMatrix[i])
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
            #print(p0Num)
            #print(p0Denom)



    #devide for all vector element
    #print(p1Num)
    #print(p1Denom)
    #let each element in p1Num vector to devide / p1Denom => each element has its own p(word_i | category_i)
    #p1Vect = np.log(p1Num/p1Denom)          #change to log()
    #p0Vect = np.log(p0Num/p0Denom)          #change to log()
    p1Vect = (p1Num/p1Denom)          #change to log()
    p0Vect = (p0Num/p0Denom)          #change to log()
    print(p1Vect)
    print(p0Vect)
    return p0Vect,p1Vect,pAbusive


def test():
    listOPosts,classVec=loadDataSet()
    #print(listOPosts,classVec)
    myVocabList=createVocabList(listOPosts)
    print(myVocabList)

    #try to change 1 doc to vector
    print(listOPosts[0])
    #by word appear or not.
    returnVec=setOfWords2Vec(myVocabList,listOPosts[0])

    #by word frequence.
    #returnVec=bagOfWords2VecMN(myVocabList,listOPosts[0])
    print(returnVec)


    return 0


def test2():
    listOPosts,listClasses=loadDataSet()
    #print(listOPosts,listClasses)
    myVocabList=createVocabList(listOPosts)
    print(myVocabList)

    #change all data word-> 0/1 vector
    trainMat=[]
    for postingDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postingDoc))


    #print(trainMat)
    #print(listClasses)

    p0V,p1V,pAb=trainNB0_original(trainMat,listClasses)
    #print(p0V)
    #print(p1V)
    #print(pAb)

    return 0

if __name__ == "__main__":
    test()
    #test2()



