__author__ = 'jurre'
import os, glob
import csv
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.metrics import hamming_loss
from sklearn.model_selection import cross_val_score
from numpy import genfromtxt

def listToMat(l):
    return np.array(l).reshape(-1, np.array(l).shape[-1])


def csvFormatedOutput(fileName, ans):
    with open(fileName, 'wb') as f:
        c = csv.writer(f, delimiter=',')

        c.writerow(['ID','Sample','Label','Predicted'])
        tstring = ['gender', 'age', 'health']
        counter = 0
        for i in range(0, len(ans)):
            for j in range(0, 3):
                c.writerow([counter, i, tstring[j],  ans[i,j] ])
                counter +=1
    return 0

def CVsc(classifier, X, Y, cv = 10):
    hamloss = make_scorer(hamming_loss, greater_is_better=False)
    return(cross_val_score(classifier, X, Y, cv = cv, scoring=hamloss))

def loadData():

    trainTarget = "data/targets.csv"
    temp = []
    temp.append(genfromtxt(trainTarget, delimiter=','))
    trainingTargets = temp[0]


    #read the training features
    trainFeature = 'features/trainFeatureHistR.csv'
    temp = []
    temp.append(genfromtxt(trainFeature, delimiter=','))
    trainingFeatures = (listToMat(temp))

    # read the testing features
    testFeature = 'features/testFeatureHistR.csv'
    temp = []
    temp.append(genfromtxt(testFeature, delimiter=','))
    testingFeatures = (listToMat(temp))

    return(trainingTargets, trainingFeatures, testingFeatures)