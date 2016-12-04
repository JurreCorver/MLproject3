import csv
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import multiclass

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


def adaBoostPredict(boostModels, boostAlpha, boost, XTest):
    predicted = np.zeros([XTest.shape[0], ])
    predictedTemp = np.zeros([XTest.shape[0], ])

    for i in range(boost):
        predictedTemp += boostAlpha[i] * boostModels[i].predict(XTest)

    predicted = predicted / np.sum(boostAlpha)  # normalization

    for i in range((XTest.shape[0])):
        if predictedTemp[i] < 0.5:
            predicted[i] = int(0)
        else:
            predicted[i] = int(1)
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
trainFeature = '../features/trainFeature.csv'
temp = []
temp.append(genfromtxt(trainFeature, delimiter=','))
trainingFeatures = (listToMat(temp)).T

# read the testing features
testFeature = '../features/testFeature.csv'
temp = []
temp.append(genfromtxt(testFeature, delimiter=','))
testingFeatures = (listToMat(temp)).T

# reformat the training data
x = []
y = []
for i in range(trainingFeatures.shape[0]):
    for j in range(trainingTargets.shape[1]):
        if trainingTargets[i,j] ==1:
            x.append(trainingFeatures[i,:])
            y.append(j)
trainXInput = listToMat(x)
trainYInput = np.reshape(listToMat(y),[-1, ])

# multiclass one vs one
sgdBase = SGDClassifier(alpha=0.1, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='squared_hinge', n_iter=1000,
       n_jobs=1, penalty='l1', power_t=0.5, random_state=None,
       shuffle=True, verbose=0, warm_start=False)
multiOvR = multiclass.OneVsRestClassifier(sgdBase)

# sgd + adaboost
numBoost = 1000
# multiOvO.fit(trainXInput, trainYInput)
boostEps, boostAlp, boostModel = adaBoost(multiOvR, numBoost,
                                          trainXInput, trainYInput)
print(boostAlp)
print(boostModel)

# prediction
nClasses = trainingTargets.shape[1]
boundaryVal = 0.33

# defined output

baseFileName = '../multiOvRSgd'
outBoost = range(100, numBoost+1, 100)
for i in range(len(outBoost)):
    predicted = adaBoostWeightVoting(boostModel, boostAlp, outBoost[i], testingFeatures,
                         nClasses, boundaryVal)
    csvMLP3FormatedOutput(baseFileName + str(i)+'.csv', predicted)
