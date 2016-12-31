Part I Team Information
Team name: 
Tschick
Team member: 
Jurre Corver(Legi Nr.: 15-945-090), 
Mei-Chih Chang (Legi Nr.: 09-909-680), 
Caifa Zhou (Legi Nr.: 15-900-293)

Part II: General Model
The general procedures of the proposed the model for the age prediction from the brain fMRI consist of:
(i) feature extraction
Using the nibabel/NIfTI tools to read the intensity of fMRI image, and take each 3D fMRI as a data point. Blocklizing each fMRI by 0 to 3 levels. It means that we blocklize the 3D fMRI to 8*8*8 blocks by dichotimizing along the 3 axis for 3 times. According to the intensity of each block, we proposed to compute the histogram features with 64 bins for each block. In total there are (1 + 8 + 8*8 + 8*8*8)*64 (in total is 37440) features extracted for each training and testing fMRI

(ii) multiclassification model
Applying LASSO with LARS (LASSOLARS) using 10 fold cross validation with maximal 5000 times iteration for search for the optimal hyperparameter, i.e. lambda, for the model. We used the sklearn package.

(iii) integerization and cutoff the regression result
Base model: using the logistic regression based classifer as the base classifer of oneVsRestclassifer from sklearn package. The logistc regression classifer applies 10 fold cross validation. 

Pipeline: To adapt the model to the training dataset, we apply the pipeline model to the base classifer using the standard scaler.


