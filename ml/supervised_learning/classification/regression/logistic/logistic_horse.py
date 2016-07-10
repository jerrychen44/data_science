'''
@author: Peter
@modifier: Jerry
'''

import numpy as np
import os,sys
import logistic_reg as logistic

filepath=os.path.dirname(os.path.realpath(__file__))

def classifyVector(inX, weights):
    #print(inX)
    prob = logistic.sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open(filepath+'/data_set/horseColicTraining.txt')
    frTest = open(filepath+'/data_set/horseColicTest.txt')

    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))


    trainWeights = logistic.stocGradAscent1(np.array(trainingSet), trainingLabels, 1000)

    #test part
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        #to 21 is because data with 22 features
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1

    errorRate = (float(errorCount)/numTestVec)
    print ("the error rate of this test is: %f" % errorRate)
    return errorRate



def multiTest():
    numTests = 10
    errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()

    print ("after %d iterations the average error rate is: %f" \
            % (numTests, errorSum/float(numTests)) )
    return 0


def test():
    errorRate=colicTest()
    print(errorRate)
    return 0

if __name__=='__main__':
    test()
    #multiTest()
