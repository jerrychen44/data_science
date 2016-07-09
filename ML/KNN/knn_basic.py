'''
kNN: k Nearest Neighbors

Input:      input_data_X: vector to compare to existing dataset (1xN), type is list
            training_dataset: size m data set of known vectors (NxM) (training data set)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)

Output:     the most popular class label

Source from: Peter,stack-overflow.
Modify,improve,integration: Jerrychen
'''

#from numpy import *
import numpy as np
import operator
from os import listdir
import matplotlib.pyplot as plt


def plot_xy_scatter(dataset,labels,testX):
    fig=plt.figure()
    ax = fig.add_subplot(111)
    colors = ['b', 'c', 'y', 'm', 'r']

    #print(dataset[0][0])
    #print(dataset[:,0] ,dataset[:,1])

    #ax.scatter(X,Y)
    ax.scatter(dataset[:,0],dataset[:,1],marker='x', color=colors[0])
    #print(type(dataset),type(labels))

    for index  in range(len(dataset.tolist())):
        ax.text(dataset.tolist()[index][0], dataset.tolist()[index][1],labels[index] )


    ax.scatter(testX[0],testX[1],marker='o', color=colors[4])
    ax.text(testX[0], testX[1],testX[2] )
    plt.show()
    return 0


def createDataSet():
    #each data has two feature now.
    training_dataset = np.array([[10.0,10.1],[7.0,8.0],[3,3],[3,5.1],[6,4.1],[2.8,5.1]])
    #each Y of above data
    labels = ['A','A','B','B','C','B']
    return training_dataset, labels

#input_data_X : input point. (testing data)
#data set
#y : training_dataset
#k : check k points (k neighbors) near by input_data_X , and will choice voted most one.
def classify0(input_data_X, training_dataset, labels, k):

    #####################
    #count distance     #
    #Euclidean distance #
    #####################
    dataSetSize = training_dataset.shape[0]
    #print(dataSetSize)

    #print(input_data_X,(dataSetSize,1))
    #print(tile(input_data_X, (dataSetSize,1)))
    #print(input_data_X)
    #print(training_dataset)

    #tile(A, reps)
    #Construct an array by repeating A the number of times given by reps.
    #because we try to do the distance by martix way.
    #do "-" to each coresponding postion on matrix.
    diffMat = np.tile(input_data_X, (dataSetSize,1)) - training_dataset
    #print(diffMat)
    sqDiffMat = diffMat**2
    #print(sqDiffMat)
    #sum(axis=1) means sum the martix item with x direction
    sqDistances = sqDiffMat.sum(axis=1)
    #print(sqDistances)
    #the item of sqDistances is distances of the inputX between training_dataset before **0.5
    distances = sqDistances**0.5
    #distance matrix has all data set Euclidean distance between inputX, we will find minized larter.
    #print(distances)

    #sort distance from small to large. (but we just recourd the order map to sortedDistIndicies. not really change the order from distance)
    sortedDistIndicies = distances.argsort()
    #print(sortedDistIndicies)


    #############################################
    #classify by the number k of nearest points #
    #############################################
    classCount={}
    #iterate k times. find out K item around input X. and select out which one is voted most.
    #ex: if k = 1 ,means, find the nearest point by the input X directly, and use its label be the input X classify result.
    #classCount is to records the what class for k data from sortedDistIndicies, and record the number for its dic value
    #ex : {'A': 2, 'B': 1}
    #print(sortedDistIndicies[0])
    for i in range(k):
        #we find out the Y value from the distance smallest item.
        voteIlabel = labels[sortedDistIndicies[i]]


        #dict.get(key, default=None)
        #key=This is the Key to be searched in the dictionary.
        #default -- This is the Value to be returned in case key does not exist.
        #print(classCount)
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        #print(voteIlabel,classCount.get(voteIlabel,0),classCount[voteIlabel])
        #print(classCount)


    #sorted, put the class which has largest count at first, and we judge the inputX is much more close
    #this class (due to it has largest count.)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #print(sortedClassCount)

    #[('A', 2), ('B', 1)]

    #extract out the class by string.
    return sortedClassCount[0][0]

def test_classify0():
    training_dataset, labels=createDataSet()
    #you can set the different point x, y to see belong A or B
    testX=[7.2,3]
    print(type(testX),type(training_dataset),type(labels))
    ret=classify0(testX, training_dataset, labels, 3)
    print(ret)
    testX.append(ret)
    plot_xy_scatter(training_dataset,labels,testX)
    return 0

if __name__ == "__main__":
    test_classify0()


