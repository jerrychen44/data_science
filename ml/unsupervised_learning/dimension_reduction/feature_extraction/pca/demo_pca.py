# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import os

from matplotlib import pylab
import pandas as pd
import numpy as np

from sklearn import linear_model, decomposition
from sklearn import lda

logistic = linear_model.LogisticRegression()


from utils import CHART_DIR,DATA_DIR

np.random.seed(3)


def plot_simple_demo_1():
    pylab.clf()
    fig = pylab.figure(num=None, figsize=(10, 4))
    pylab.subplot(121)

    title = "Original feature space"
    pylab.title(title)
    pylab.xlabel("$X_1$")
    pylab.ylabel("$X_2$")

    x1 = np.arange(0, 10, .2)
    x2 = x1 + np.random.normal(scale=1, size=len(x1))

    good = (x1 > 5) | (x2 > 5)
    bad = ~good

    x1g = x1[good]
    x2g = x2[good]
    pylab.scatter(x1g, x2g, edgecolor="blue", facecolor="blue")

    x1b = x1[bad]
    x2b = x2[bad]
    pylab.scatter(x1b, x2b, edgecolor="red", facecolor="white")

    pylab.grid(True)


    ######################################
    pylab.subplot(122)

    X = np.c_[(x1, x2)]
    np.savetxt(DATA_DIR+"/data_set_demo_simple.csv", X, delimiter=",")

    pca = decomposition.PCA(n_components=1)
    Xtrans = pca.fit_transform(X)

    Xg = Xtrans[good]
    Xb = Xtrans[bad]

    pylab.scatter(
        Xg[:, 0], np.zeros(len(Xg)), edgecolor="blue", facecolor="blue")
    pylab.scatter(
        Xb[:, 0], np.zeros(len(Xb)), edgecolor="red", facecolor="white")
    title = "Transformed feature space"
    pylab.title(title)
    pylab.xlabel("$X'$")
    fig.axes[1].get_yaxis().set_visible(False)

    print(pca.explained_variance_ratio_)

    pylab.grid(True)

    pylab.autoscale(tight=True)
    filename = "pca_demo_simple.png"
    pylab.savefig(os.path.join(CHART_DIR, filename), bbox_inches="tight")
    pylab.show()


def plot_simple_demo_2():
    pylab.clf()
    fig = pylab.figure(num=None, figsize=(10, 4))
    pylab.subplot(121)

    title = "Original feature space"
    pylab.title(title)
    pylab.xlabel("$X_1$")
    pylab.ylabel("$X_2$")

    x1 = np.arange(0, 10, .2)
    x2 = x1 + np.random.normal(scale=1, size=len(x1))

    good = x1 > x2
    bad = ~good

    x1g = x1[good]
    x2g = x2[good]
    pylab.scatter(x1g, x2g, edgecolor="blue", facecolor="blue")

    x1b = x1[bad]
    x2b = x2[bad]
    pylab.scatter(x1b, x2b, edgecolor="red", facecolor="white")

    pylab.grid(True)


    ######################################
    pylab.subplot(122)

    X = np.c_[(x1, x2)]
    np.savetxt(DATA_DIR+"/data_set_demo_complex.csv", X, delimiter=",")
    #print(X)
    pca = decomposition.PCA(n_components=1)
    Xtrans = pca.fit_transform(X)

    Xg = Xtrans[good]
    Xb = Xtrans[bad]

    pylab.scatter(
        Xg[:, 0], np.zeros(len(Xg)), edgecolor="blue", facecolor="blue")
    pylab.scatter(
        Xb[:, 0], np.zeros(len(Xb)), edgecolor="red", facecolor="white")
    title = "Transformed feature space"
    pylab.title(title)
    pylab.xlabel("$X'$")
    fig.axes[1].get_yaxis().set_visible(False)

    print(pca.explained_variance_ratio_)

    pylab.grid(True)

    pylab.autoscale(tight=True)
    filename = "pca_demo_complex.png"
    pylab.savefig(os.path.join(CHART_DIR, filename), bbox_inches="tight")
    pylab.show()

# LDA=Linear Discriminant Analyisis
def plot_simple_demo_lda():
    pylab.clf()
    fig = pylab.figure(num=None, figsize=(10, 4))
    pylab.subplot(121)

    title = "Original feature space"
    pylab.title(title)
    pylab.xlabel("$X_1$")
    pylab.ylabel("$X_2$")

    #data set
    x1 = np.arange(0, 10, .2)
    x2 = x1 + np.random.normal(scale=1, size=len(x1))
    #save to disk
    #X = np.c_[(x1, x2)]
    #print(X,type(X))


    good = x1 > x2
    bad = ~good

    x1g = x1[good]
    x2g = x2[good]
    pylab.scatter(x1g, x2g, edgecolor="blue", facecolor="blue")

    x1b = x1[bad]
    x2b = x2[bad]
    pylab.scatter(x1b, x2b, edgecolor="red", facecolor="white")

    pylab.grid(True)


    ######################################
    pylab.subplot(122)

    X = np.c_[(x1, x2)]
    np.savetxt(DATA_DIR+"/data_set_lda.csv", X, delimiter=",")

    lda_inst = lda.LDA(n_components=1)
    Xtrans = lda_inst.fit_transform(X, good)

    Xg = Xtrans[good]
    Xb = Xtrans[bad]

    pylab.scatter(
        Xg[:, 0], np.zeros(len(Xg)), edgecolor="blue", facecolor="blue")
    pylab.scatter(
        Xb[:, 0], np.zeros(len(Xb)), edgecolor="red", facecolor="white")
    title = "Transformed feature space"
    pylab.title(title)
    pylab.xlabel("$X'$")
    fig.axes[1].get_yaxis().set_visible(False)

    pylab.grid(True)

    pylab.autoscale(tight=True)
    filename = "lda_demo_lda.png"
    pylab.savefig(os.path.join(CHART_DIR, filename), bbox_inches="tight")
    pylab.show()

if __name__ == '__main__':
    #pass
    plot_simple_demo_1()
    plot_simple_demo_2()
    plot_simple_demo_lda()
