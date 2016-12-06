__author__ = 'jurre'

from jsrc.functions import *

import numpy as np
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline

#import the features - make sure to change the paths in functions.py
trainingTargets, trainingFeatures, testingFeatures = loadData()

#rename for convenience of brevity
X = trainingFeatures
Y = trainingTargets
Z = testingFeatures

#this is our loss function, which we want to minimize and no maximize. We use the OnevsRest classifier
hamloss = make_scorer(hamming_loss, greater_is_better=False)
classif2 = OneVsRestClassifier(LogisticRegressionCV(Cs=10, fit_intercept=True, cv=10, penalty='l2', scoring=hamloss))

#do some linear scaling before fitting the data
pipe2 = make_pipeline(StandardScaler(), classif2)
pipe2.fit(X,Y)


#performance can be evaluated by cross_val_score: np.mean(CVsc(pipe2, X, Y, cv = 10))
#write answer to file
ans2 = pipe2.predict(Z)
csvFormatedOutput('jsrc/jsub.csv', ans2)