'''
Source from: Peter,stack-overflow.
Modify,improve,integration: Jerrychen
'''

import numpy as np
import bayes_basic_v1_before_log_p1p0 as bbv1


def trainNB0(trainMatrix,trainCategory):
    #initial probability
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    #print(numTrainDocs,numWords)
    #sum(trainCategory) is to count how many of 1.
    #pAbusive = (total abusive doc /total number of data) => p(abusive) =>p(category_i)
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    #print(sum(trainCategory),pAbusive)
    #p0Num = np.zeros(numWords)
    #p1Num = np.zeros(numWords)
    #change to ones() because originally will get 0 probability, later on, multiply them will just got 0.
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    #print(p0Num)

    #pXDenom is the total count of word appears number.
    #p0Denom is for category 0, not abusive doc, after go through them, total appears count.0
    #p1Denom is whole appeared words count in category 1.

    #p0Denom = 0.0
    #p1Denom = 0.0
    #change to 2.0
    p0Denom = 2.0
    p1Denom = 2.0

    #print(trainCategory)
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
    p1Vect = np.log(p1Num/p1Denom)          #change to log() because truncate of too small multiply reult.P(w1|ci)*P(w2|ci).....
    p0Vect = np.log(p0Num/p0Denom)          #change to log()
    #p1Vect = (p1Num/p1Denom)          #original probability
    #p0Vect = (p0Num/p0Denom)
    #print(p1Vect)
    #print(p0Vect)
    #print(pAbusive)
    return p0Vect,p1Vect,pAbusive


def test():
    listOPosts,listClasses=bbv1.loadDataSet()
    #print(listOPosts,listClasses)
    myVocabList=bbv1.createVocabList(listOPosts)
    #print(myVocabList)

    #change all data word-> 0/1 vector
    trainMat=[]
    for postingDoc in listOPosts:
        #just use the word appear of not.
        trainMat.append(bbv1.setOfWords2Vec(myVocabList,postingDoc))

        #trying to sue bag of words
        #trainMat.append(bbv1.bagOfWords2VecMN(myVocabList,postingDoc))



    #print(trainMat)
    #print(listClasses)

    p0V,p1V,pAb=trainNB0(trainMat,listClasses)
    print(p0V)
    print(p1V)
    print(pAb)

    return 0

if __name__ == "__main__":
    test()



