__author__ = 'jurre'

from src.functions import *

import numpy as np
import matplotlib.pyplot as plt

from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
#from src.fMRIToFeature import *

#Here we load the Data.
#@Caifa: This will probably not work on your computer as I am still on an old python version. Just change the function
# mriToHistFeature in src.fMRIToFeature to your importing function and it will work
'''bins = 64
block = 8
testDir = 'data/set_test/'
trainDir = 'data/set_train/'

featureTest = mriToHistFeature(testDir, bins, block)
csvOutput('src/testFeatureHistR.csv', featureTest)
featureTrain = mriToHistFeature(trainDir, bins, block)
csvOutput('src/trainFeatureHistR.csv', featureTrain)'''

#Here we start the logistic regression.

#load the data using the function loadData in src.functions
trainingTargets, trainingFeatures, testingFeatures = loadData()


X = trainingFeatures
Y = trainingTargets
Z = testingFeatures

hamloss = make_scorer(hamming_loss, greater_is_better=False)
classif = OneVsRestClassifier(LogisticRegressionCV(Cs=10, fit_intercept=True, cv=10, penalty='l2' , scoring=hamloss)) #solver='newton-cg'

pipe = make_pipeline(StandardScaler(), classif)
pipe.fit(X,Y)

anss = pipe.predict(Z)
csvFormatedOutput('final_sub.csv', anss)