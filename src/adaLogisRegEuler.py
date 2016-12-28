#!/usr/bin/env python3

import csv
import numpy as np
from numpy import genfromtxt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import SGDClassifier, LassoLarsCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
import sys

# csv write a formated csv file
def csvFormatedOutput(fileName, ans):
    with open(fileName, 'w', newline='') as f:
        c = csv.writer(f, delimiter=',',
                       quotechar='|', quoting=csv.QUOTE_MINIMAL)
        c.writerow(['ID', 'Prediction'])
        for i in range(0, len(ans)):
            #             temp = listToMat()
            c.writerow([i + 1, ans[i]])
    return 0


def listToMat(l):
    return np.array(l).reshape(-1, np.array(l).shape[-1])


def plotter(measured, predicted):
    # plot the result
    fig, ax = plt.subplots()
    y = measured
    ax.scatter(y, predicted)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()
    return 0


def DimRedPlot(x, y):
    plt.figure()
    colors = ['navy', 'turquoise']
    labelName = ['bad', 'good']
    lw = 2
    # y= np.reshape(trainingTargets,[-1,])
    for color, i, label in zip(colors, [0, 1], labelName):
        plt.scatter(x[y == bin(i), 0],
                    x[(y) == bin(i), 1],
                    color=color, alpha=.8, lw=lw, label=label)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.show()
    return 0


def calibrationProb(classifier, X, y, test):
    caliClassifier = CalibratedClassifierCV(classifier, cv=10, method='isotonic')
    caliClassifier.fit(X, y)
    preTest = caliClassifier.predict(test)
    probTest = caliClassifier.predict_proba(test)
    return preTest, probTest[:, 1]


def searchClassifier(baseClassifier, paraGrid, X, y):
    clf = GridSearchCV(baseClassifier, paraGrid, cv=10,
                       scoring=make_scorer(metrics.hamming_loss))
    clf = clf.fit(X, np.reshape(y, [-1, ]))
    return clf.best_estimator_


def ldaSearchCV(f):
    ldaParaGrid = {'solver': ['eigen'],
                   'n_components': [1, 2], }
    lda = LinearDiscriminantAnalysis(shrinkage='auto')
    # ldaClf = GridSearchCV(lda, ldaParaGrid, scoring=make_scorer(metrics.accuracy_score), cv= 10)
    # ldaClf.fit(X, y)
    ldaBest = searchClassifier(lda, ldaParaGrid,
                               trainingFeatures, trainingTargets)
    ldaPreTest, ldaProbTest = calibrationProb(ldaBest,
                                              trainingFeatures, trainingTargets, testingFeatures)
    # print(ldaProbTest)
    csvFormatedOutput(f, ldaProbTest)
    return 0


def sgdSearchCV(f, trainingFeatures, trainingTargets, testingFeatures):
    sgdParaGrid = {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                   'penalty': ['l1', 'l2'],
                   'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1]}
    sgd = SGDClassifier(n_iter=1000)
    sgdBest = searchClassifier(sgd, sgdParaGrid,
                               trainingFeatures, trainingTargets)
    sgdPreTest, sgdProbTest = calibrationProb(sgdBest,
                                              trainingFeatures, trainingTargets, testingFeatures)
    csvFormatedOutput(f, sgdProbTest)
    return 0


def svcSearchCV(f, trainingFeatures, trainingTargets, testingFeatures):
    svcParaGrid = {'C': [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5],
                   'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                   'degree': [3, 4, 5, 6, 7],
                   'gamma': [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]}
    svc = SVC(probability=True)
    svcBest = searchClassifier(svc, svcParaGrid, trainingFeatures, trainingTargets)
    svcProbTest = svcBest.predict_proba(testingFeatures)
    csvFormatedOutput(f, svcProbTest[:, 1])
    return 0


def adaBoostSearchCV(baseClassifier, f, trainingFeatures, trainingTargets, testingFeatures):
    adaParaGrid = {'n_estimators': [10, 50, 100, 200],
                   'learning_rate': [0.001, 0.01, 0.1, 0.5, 1],
                   'algorithm': ['SAMME'], }
    ada = AdaBoostClassifier(base_estimator=baseClassifier, random_state=0)
    adaBest = searchClassifier(ada, adaParaGrid, trainingFeatures, trainingTargets)
    adaPreTest, adaProbTest = calibrationProb(adaBest, trainingFeatures, trainingTargets,
                                              testingFeatures)
    csvFormatedOutput(f, adaProbTest, trainingFeatures, trainingTargets, testingFeatures)


def baggingSearchCV(baseClassifier, f):
    baggingParaGrid = {'n_estimators': [10, 50, 100, 200]}
    bagging = BaggingClassifier(base_estimator=baseClassifier, max_features=1.0, max_samples=1.0)
    baggingBest = searchClassifier(bagging, baggingParaGrid, trainingFeatures, trainingTargets)
    baggingPreTest, baggingProbTest = calibrationProb(baggingBest,
                                                      trainingFeatures, trainingTargets, testingFeatures)
    csvFormatedOutput(f, baggingProbTest)


def computeEpsAlp(yTrue, yTrainPred, initWeights):
    temp = []
    for i in range(len(yTrue)):
        if (yTrue[i] != yTrainPred[i]):
            temp.append(initWeights[i])
    epsilon = np.sum(temp) / np.sum(initWeights)
    alpha = np.log((1 - epsilon) / (epsilon + 1e-10))
    for i in range(len(yTrue)):
        if (yTrue[i] != yTrainPred[i]):
            initWeights[i] = initWeights[i] * np.exp(alpha)
    return epsilon, alpha, initWeights


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


def adaBoost(sgdBase, numBoost, XTrain, yTrain):
    boostModel = []
    boostAlp = []
    boostEps = []
    numTrain = yTrain.shape[0]
    initWeights = []
    for i in range(numTrain):
        initWeights.append(1 / float(numTrain))

    for i in range(numBoost):
        calibCV = CalibratedClassifierCV(sgdBase, cv=10, method='isotonic')
        calibCV.fit(XTrain, yTrain, sample_weight=np.reshape(initWeights, [-1, ]))
        boostModel.append(calibCV)
        x, y, initWeights = computeEpsAlp(yTrain, calibCV.predict(XTrain), initWeights)
        boostEps.append(x)
        boostAlp.append(y)
    return boostEps, boostAlp, boostModel


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


def adaBoostPredict(boostModels, boostAlpha, boost, XTest):
    predicted = np.zeros([XTest.shape[0], ])
    predictedTemp = np.zeros([XTest.shape[0], ])

    for i in range(boost):
        predictedTemp += boostAlpha[i] * boostModels[i].predict(XTest)

    predictedTemp = predictedTemp / np.sum(boostAlpha)  # normalization

    for i in range((XTest.shape[0])):
        if predictedTemp[i] < 0.5:
            predicted[i, j] = int(0)
        else:
            predicted[i, j] = int(1)
    return predicted


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


def adaBoostProbaPredict(boostModels, boostAlpha, boost, XTest, predicted):
    predictedProba = np.zeros([XTest.shape[0], 2])
    predictedProbaTemp = np.zeros([XTest.shape[0], 1])

    # initilization
    for i in range(XTest.shape[0]):
        predictedProbaTemp[i, 0] = 1

    # normalized alpha
    boostAlpha /= np.sum(boostAlpha)

    for i in range(boost):
        temp = boostModels[i].predict(XTest)
        tempProba = boostModels[i].predict_proba(XTest)
        for j in range(len(temp)):
            predictedProbaTemp[j, 0] *= tempProba[j, temp[j]]

    # assign the proba
    for i in range(len(predicted)):
        if predicted[i,] == 0:
            predictedProba[i, 0] = predictedProbaTemp[i, 0]
            predictedProba[i, 1] = 1 - predictedProbaTemp[i, 0]
        else:
            predictedProba[i, 0] = 1 - predictedProbaTemp[i, 0]
            predictedProba[i, 1] = predictedProbaTemp[i, 0]
    return predictedProba


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


def adaBoostWeightVoting(boostModel, boostAlp, numBoost, testingFeatures,
                         nClasses, boundaryVal):
    # prediction of adaboost
    nTest = testingFeatures.shape[0]
    predicted = np.zeros([nTest, nClasses])
    boostPred = np.zeros([nTest, numBoost])
    voteWeights = boostAlp[0:numBoost] / np.sum(boostAlp[0:numBoost])
    for i in range(numBoost):
        boostPred[:, i] = boostModel[i].predict(testingFeatures)

    # weighted voting
    for i in range(nTest):
        for j in range(nClasses):
            predicted[i, j] = np.sum(voteWeights[boostPred[i, :] == j])

    # decison boundary

    predicted[predicted >= boundaryVal] = 1
    predicted[predicted < boundaryVal] = 0
    return predicted

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

# main begins
# load data
# read the training targets
trainTarget = '../data/targets.csv'
temp = []
temp.append(genfromtxt(trainTarget, delimiter=','))
trainingTemp = (listToMat(temp)).T
temp = []
for i in range(trainingTemp.shape[0]):
    # if trainingTemp[i] == 0:
    #     trainingTemp[i] = -1
    temp.append(((trainingTemp[i])))
trainingTargets = listToMat(temp).T

# read the training features
trainFeature = '../features/trainFeatureHistR.csv'
temp = []
temp.append(genfromtxt(trainFeature, delimiter=','))
trainingFeatures = (listToMat(temp))

# read the testing features
testFeature = '../features/testFeatureHistR.csv'
temp = []
temp.append(genfromtxt(testFeature, delimiter=','))
testingFeatures = (listToMat(temp))

hamloss = metrics.make_scorer(metrics.hamming_loss, greater_is_better=False)
classif2 = OneVsRestClassifier(LogisticRegressionCV(Cs=10, fit_intercept=True, cv=10, penalty='l2', scoring=hamloss))
# classif2 = OneVsRestClassifier(LassoLarsCV(max_iter=2000, max_n_alphas=1000, cv = 10))
#do some linear scaling before fitting the data
pipe2 = make_pipeline(StandardScaler(), classif2)

# ada
nClf = trainingTargets.shape[1]
nBoost = np.int(sys.argv[1])
boostEps, boostAlp, boostModel = adaMultiOutBoost(pipe2, nBoost, trainingFeatures, trainingTargets)
adaPredicted = adaBoostMultiPredict(boostModel, boostAlp, nBoost, testingFeatures, nClf)

fileName = "../OvRAdaLogRegHistR" + str(nBoost) + ".csv"
csvMLP3FormatedOutput(fileName, adaPredicted)