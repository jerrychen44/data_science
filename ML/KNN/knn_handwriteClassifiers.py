'''
Source from: Peter,stack-overflow.
Modify,improve,integration: Jerrychen
'''
import numpy as np
import operator,os
import pandas as pd
import matplotlib.pyplot as plt
import knn_basic as knnb

filepath=os.path.dirname(os.path.realpath(__file__))#root


def img2vector(filename):
    #x_xxx.txt is an 32x32 bit map.
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    #to 32 is because the row number in traingdata x_xxx.txt is 32.is
    #go through each row.
    for i in range(32):
        lineStr = fr.readline()
        #go through each column.
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])

    #print to chceck
    #np.set_printoptions(threshold=np.nan,linewidth=200)
    #print(returnVect)
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir(filepath+'/data_set/digits/trainingDigits')           #load the training set

    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    #print(len(trainingFileList),len(trainingMat),m)

    #go through all traing files. x_xx.txt
    #want to crerate a traing matrix, each row is a traing data,feature x is the 0 or 1 for each pixel
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector(filepath+'/data_set/digits/trainingDigits/%s' % fileNameStr)


    #np.set_printoptions(threshold=np.nan,linewidth=200)
    #print(trainingMat[0])

    #find the test set
    testFileList = os.listdir(filepath+'/data_set/digits/testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(filepath+'/data_set/digits/testDigits/%s' % fileNameStr)
        classifierResult = knnb.classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print ("%s the classifier came back with: %d, the real answer is: %d" % (fileNameStr,classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
            print("=======ERROR")

    print ("\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))
    return 0

if __name__ == "__main__":
    handwritingClassTest()

    #img2vector(filepath+'/data_set/digits/trainingDigits/8_61.txt')

