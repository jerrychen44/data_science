'''
@author: Peter
@modifier: Jerry
'''

import numpy as np
import os
import pprint
#root path
filepath=os.path.dirname(os.path.realpath(__file__))

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open(filepath+'/data_set/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        #we set X0=1,and feature X1=float(lineArr[0]), X2=float(lineArr[1])
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    #pprint.pprint(dataMat)
    #pprint.pprint(labelMat)
    return dataMat,labelMat

def sigmoid(inX):
    #print(inX)
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    #convert dataMatIn to NumPy matrix
    dataMatrix = np.mat(dataMatIn)
    print(np.shape(dataMatrix))
    #convert classLabels to NumPy matrix, and transpose
    labelMat = np.mat(classLabels).transpose()
    print(np.shape(labelMat))

    # m data,n features
    m,n = np.shape(dataMatrix)
    #learing step rate
    alpha = 0.001
    #iteration number
    maxCycles = 500
    #initial weights vector to all 1.
    #weight = theta in Andrew Ng class
    weights = np.ones((n,1))
    print(np.shape(weights))

    #quick test
    #print(dataMatrix)
    #print(weights)
    #print(dataMatrix*weights)
    #print(np.shape(dataMatrix*weights))
    #print(labelMat[0:5])
    #h=sigmoid(dataMatrix*weights)
    #print(h[0:5])
    #error = (labelMat - h)
    #print(error[0:5])


    # * is matrix mult operation
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #matrix mult
        #error is the error between real label 1/0 probability and sigmoid output probability
        error = (labelMat - h)              #vector subtraction

        # + is gradAscent, - is gradDescent
        # gradent is the item dataMatrix.transpose()* error
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult

    return weights

def plotBestFit(weights,callfrom='gradAscent'):
    import matplotlib.pyplot as plt
    #print(type(weights),type(weights.getA()))
    #change type matrix to ndarray for plot
    if callfrom =='gradAscent':
        weights = weights.getA()
    dataMat,labelMat=loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

#gradAscent_random init
def stocGradAscent0(dataMatrix, classLabels):
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights
#improved random GradAscent1
#because Andrew Ng's J(theta)=- (1/m)l(n), so he use gradent Descent to find minimum.
def stocGradAscent1(dataMatrix, classLabels, numIter=500):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
            randIndex = int(np.random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def test_gradAscent():
    dataMat,labelMat=loadDataSet()
    #pprint.pprint(dataMat)
    #=======================================
    #weights=gradAscent(dataMat, labelMat)
    #plotBestFit(weights)

    #======================================
    weights=stocGradAscent1(np.array(dataMat), labelMat)
    print(weights)
    plotBestFit(weights,'stocGradAscent0')
    #======================================

    return 0

if __name__=='__main__':
    test_gradAscent()



