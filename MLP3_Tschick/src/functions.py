from sklearn import metrics

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

    trainTarget = "src/targets.csv"
    temp = []
    temp.append(genfromtxt(trainTarget, delimiter=','))
    trainingTargets = temp[0]


    #read the training features
    trainFeature = 'src/trainFeatureHistR.csv'
    temp = []
    temp.append(genfromtxt(trainFeature, delimiter=','))
    trainingFeatures = (listToMat(temp))

    # read the testing features
    testFeature = 'src/testFeatureHistR.csv'
    temp = []
    temp.append(genfromtxt(testFeature, delimiter=','))
    testingFeatures = (listToMat(temp))

    return(trainingTargets, trainingFeatures, testingFeatures)

def weightedRandomGeneration(Xtrain, ytrain, weights):
#     minSelSample = int(np.sqrt(Xtrain.shape[0]) + 1)
    minSelSample = 100
    maxSelSample = int(Xtrain.shape[0])
    selSample = np.random.choice(np.arange(minSelSample, maxSelSample),
                                 size=1, replace='false')
    selIndex = np.random.choice(Xtrain.shape[0], size=selSample, replace='false', p=weights)
    XSel = Xtrain[selIndex][:]
    ySel = ytrain[selIndex][:]
    return XSel, ySel

def csvMLP3FormatedOutput(fileName, ans):
    label = []
    label.append('gender')
    label.append('age')
    label.append('health')

    with open(fileName, 'w', newline='') as f:
        c = csv.writer(f, delimiter=',',
                       quotechar='|', quoting=csv.QUOTE_MINIMAL)
        c.writerow(['ID', 'Sample', 'Label', 'Predicted'])
        for i in range(0, len(ans)):
            for j in range(len(ans[i])):
                if ans[i][j] == 1:
                    c.writerow([3 * i + j, i, label[j], 'True'])
                else:
                    c.writerow([3 * i + j, i, label[j], 'False'])
    return 0

def adaBoostMultiPredict(boostModels, boostAlpha, boost, XTest, nClasses):
    predicted = np.zeros([XTest.shape[0], nClasses])
    predictedTemp = np.zeros([XTest.shape[0], nClasses])

    weights = boostAlpha[0:boost] / np.sum(boostAlpha[0:boost])
    for i in range(boost):
        for j in range(nClasses):
            predictedTemp[:, j] += weights[i] * boostModels[i].predict(XTest)[:, j]
    for i in range((XTest.shape[0])):
        for j in range(nClasses):
            if predictedTemp[i, j] < 0.5:
                predicted[i, j] = int(0)
            else:
                predicted[i, j] = int(1)
    return predicted

def adaMultiOutBoost(base, numBoost, XTrain, yTrain):
    boostModel = []
    boostAlp = []
    boostEps = []
    numTrain = yTrain.shape[0]
    initWeights = []
    for i in range(numTrain):
        initWeights.append(1 / float(numTrain))

    for i in range(numBoost):
        XSel,ySel = weightedRandomGeneration(XTrain, yTrain, initWeights)
        base.fit(XSel, ySel)
        boostModel.append(base)
        x, y, initWeights = computeMultiEpsAlp(yTrain, base.predict(XTrain), initWeights)
        boostEps.append(x)
        boostAlp.append(y)
    return boostEps, boostAlp, boostModel

def computeMultiEpsAlp(yTrue, yTrainPred, initWeights):
    temp = []
    for i in range(len(yTrue)):
        if (metrics.hamming_loss(yTrue[i, :], yTrainPred[i, :]) != 0):
            temp.append(initWeights[i])
    epsilon = np.sum(temp) / np.sum(initWeights)
    alpha = np.log((1 - epsilon) / (epsilon + 1e-10))
    for i in range(len(yTrue)):
        hamD = metrics.hamming_loss(yTrue[i, :], yTrainPred[i, :])
        if (hamD != 0):
            initWeights[i] = hamD * initWeights[i] * np.exp(alpha)
    initWeights = initWeights/np.sum(initWeights)
    return epsilon, alpha, initWeights
