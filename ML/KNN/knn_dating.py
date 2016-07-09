'''
Source from: Peter,stack-overflow.
Modify,improve,integration: Jerrychen
'''
#import numpy as np
import numpy as np
import operator,os
import pandas as pd
import matplotlib.pyplot as plt
import knn_basic as knnb

filepath=os.path.dirname(os.path.realpath(__file__))#root

#load datingTestSet.txt to matrix to use.
def file2matrix(filename):

    data_text=pd.read_table(filename,sep='\t',index_col=False,header=None)


    #data column1 = fly miles
    #data column2 = % of play video game
    #data column3 = every week icecream kilogream number.
    #data column4 = labels 1:didntLike,2:smallDoses,3:largeDoses

    #fr = open(filename)
    #print(fr.read())


    numberOfLines = data_text.shape[0]         #get the number of lines in the file
    returnMat = np.zeros((numberOfLines,3))        #prepare matrix to return, 3 = 3 columns. (because datingTestSet.txt has 3 feature for each data.)


    #covert 1:didntLike,2:smallDoses,3:largeDoses, and set result to classLabelVector
    for index, row in data_text.iterrows():

        if row[3]=='didntLike':
            data_text.loc[index,3]=1
        if row[3]=='smallDoses':
            data_text.loc[index,3]=2
        if row[3]=='largeDoses':
            data_text.loc[index,3]=3

    classLabelVector = []                       #prepare labels return
    classLabelVector=data_text[3].tolist()
    #print(classLabelVector)

    #remove the lable
    data_text.drop(data_text.columns[[3]], axis=1, inplace=True)
    #print(data_text.sort([0], ascending=[0]))


    returnMat=data_text.as_matrix()
    #print(data_text.head())
    #print((returnMat))
    #print(classLabelVector)
    return returnMat,classLabelVector
    #return 0,0

#/Users/jerrychen/Documents/github/data_science/ML/KNN/data_set

def plot_numpy_array_data(numpydata_array,labels_list):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    #x =numpydata_array column 1, play video game %, Y= icecream
    #and we put color with lables value, on each data point
    #ax.scatter(numpydata_array[:,1],numpydata_array[:,2],15.0*array(labels_list),15.0*array(labels_list))
    #plt.xlabel('Percentage of Time Spent Playing Video Games')
    #plt.ylabel('Liters of Ice Cream Consumed Per Week')
    # change other feature
    ax.scatter(numpydata_array[:,0],numpydata_array[:,1],15.0*array(labels_list),15.0*array(labels_list))

    plt.show()




    return 0

def autoNorm(dataSet):
    #print(dataSet)
    minVals = dataSet.min(0)
    #print(minVals)
    maxVals = dataSet.max(0)
    #print(maxVals)
    ranges = maxVals - minVals
    #create a new array for normDataSet later.Default is 0
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m,1))
    normDataSet = normDataSet/np.tile(ranges, (m,1))   #element wise divide
    #print(normDataSet)
    #print(ranges)
    #print(minVals)
    return normDataSet, ranges, minVals

def datingClassTest():
    data_path=filepath+'/data_set/datingTestSet.txt'
    hoRatio = 0.10      #hold out 10%
    datingDataMat,datingLabels = file2matrix(data_path)       #load data setfrom file
    #normalization
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    #count how many will be left to be testing data
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0

    #go through every each testing data
    for i in range(numTestVecs):
        #classify0(input_data_X, dataset, labels, k)
        #dataset is the data from last testing data to the end, so does lable
        #the input_data_X is the testing data feature
        classifierResult = knnb.classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],10)
        #classifierResut is 1:didntLike,2:smallDoses,3:largeDoses
        print ("the classifier came back with: " ,classifierResult)
        print ("           the real answer is: " , datingLabels[i])
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
            print("=======ERROR")
    print("error has",errorCount)
    print("total=",float(numTestVecs))
    print ("the total error rate is: " , (errorCount/float(numTestVecs)))
    return (errorCount)


def test():
    data_path=filepath+'/data_set/datingTestSet.txt'
    #data_path=filepath+'/datingTestSet.txt'
    #print(data_path)
    datingDataMat,datingLabels = file2matrix(data_path)       #load data setfrom file

    #normalized
    normMat, ranges, minVals = autoNorm(datingDataMat)

    #just take a peek of our data.
    #plot_numpy_array_data(datingDataMat,datingLabels)


    return 0

#1:didntLike,2:smallDoses,3:largeDoses
def main_classifyPersion():
    reulstList=['didntLike','smallDoses','largeDoses']
    flightMiles=1000
    percentGame=10
    icecream=0.543880


    data_path=filepath+'/data_set/datingTestSet.txt'
    datingDataMat,datingLabels = file2matrix(data_path)       #load data setfrom file
    #normalization
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #print(normMat)
    input_X=np.array([flightMiles,percentGame,icecream])


    #input also need normalize
    norminput_X=((input_X-minVals)/ranges)
    #print(norminput_X.tolist())

    #print(type(norminput_X.tolist()),type(normMat),type(datingLabels))
    classifierResult = knnb.classify0(norminput_X.tolist(),normMat,datingLabels,5)

    print(reulstList[classifierResult-1])
    return 0


if __name__ == "__main__":
    #test()

    #atingClassTest()

    main_classifyPersion()

