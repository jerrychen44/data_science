'''
@author: Peter
@modifier: Jerry
'''
from numpy import *
import os,sys
from time import sleep
import svm_simple_smo as svmsimp
import svm_smo_kernel as svmkernel
from plot import plotSupportVectors as plotSV
from plot import plotSupportVectors_RBF as plotSV_RBF
filepath=os.path.dirname(os.path.realpath(__file__))
from time import sleep

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    #load the training set
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels


def testDigits(kTup=('rbf', 10)):
    data_file=filepath+'/data_set/digits/trainingDigits'
    dataArr,labelArr = loadImages(data_file)
    b,alphas = svmkernel.smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]
    labelSV = labelMat[svInd];
    print ("there are %d Support Vectors" % shape(sVs)[0])

    #traing set testing
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = svmkernel.kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print ("the training error rate is: %f" % (float(errorCount)/m))

    #test set testing
    data_file=filepath+'/data_set/digits/testDigits'
    dataArr,labelArr = loadImages(data_file)
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = svmkernel.kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print ("the test error rate is: %f" % (float(errorCount)/m))


if __name__=='__main__':
    testDigits(('rbf',10))

