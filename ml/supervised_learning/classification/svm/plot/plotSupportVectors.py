'''
Created on Nov 22, 2010

@author: Peter
'''
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os,sys
filepath=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def plot_support_vectors(input_point_list,hand_select_new_point_list=None):

    xcord0 = []
    ycord0 = []
    xcord1 = []
    ycord1 = []
    markers =[]
    colors =[]
    fr = open(filepath+'/data_set/testSet.txt')#this file was generated by 2normalGen.py
    for line in fr.readlines():
        lineSplit = line.strip().split('\t')
        xPt = float(lineSplit[0])
        yPt = float(lineSplit[1])
        label = int(lineSplit[2])
        if (label == -1):
            xcord0.append(xPt)
            ycord0.append(yPt)
        else:
            xcord1.append(xPt)
            ycord1.append(yPt)

    fr.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord0,ycord0, marker='s', s=90)
    ax.scatter(xcord1,ycord1, marker='o', s=50, c='red')
    plt.title('Support Vectors Circled')

    for points in input_point_list:
        circle = Circle((points[0], points[1]), 0.5, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
        ax.add_patch(circle)

    if hand_select_new_point_list is not None:
        for points in hand_select_new_point_list:
            circle = Circle((points[0], points[1]), 0.1, facecolor='none', edgecolor=(0.1,0.1,0.1), linewidth=3, alpha=0.5)
            ax.add_patch(circle)
    '''
     #svm_simple_smo.py
    circle = Circle((4.6581910000000004, 3.507396), 0.5, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
    ax.add_patch(circle)
    circle = Circle((3.4570959999999999, -0.082215999999999997), 0.5, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
    ax.add_patch(circle)
    circle = Circle((6.0805730000000002, 0.41888599999999998), 0.5, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
    ax.add_patch(circle)
    circle = Circle((5.286862, -2.358286), 0.5, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
    ax.add_patch(circle)
    '''

    '''
    #svm_smo.py
    circle = Circle((3.542485, 1.977398), 0.5, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
    ax.add_patch(circle)
    circle = Circle((2.114999, -0.004466), 0.5, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
    ax.add_patch(circle)
    circle = Circle((8.127113, 1.274372), 0.5, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
    ax.add_patch(circle)
    circle = Circle((4.658191, 3.507396), 0.5, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
    ax.add_patch(circle)
    circle = Circle((8.197181, 1.545132), 0.5, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
    ax.add_patch(circle)
    circle = Circle((7.40786, -0.121961), 0.5, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
    ax.add_patch(circle)
    circle = Circle((6.960661,  -0.245353), 0.5, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
    ax.add_patch(circle)
    circle = Circle((6.080573, 0.418886), 0.5, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
    ax.add_patch(circle)
    circle = Circle((3.107511, 0.758367), 0.5, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
    ax.add_patch(circle)
    '''


    #plt.plot([2.3,8.5], [-6,6]) #seperating hyperplane
    b = -3.75567; w0=0.8065; w1=-0.2761
    x = arange(-2.0, 12.0, 0.1)
    y = (-w0*x - b)/w1
    ax.plot(x,y)
    ax.axis([-2,12,-8,6])
    plt.show()
    return 0