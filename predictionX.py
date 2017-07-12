# evaluating two classifier Random Forest and Decision Tree Classifiers
#importing required modules
import numpy as np
import pandas as pd
from sklearn import preprocessing
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import csv

#----------------------------------------------------------------
# DATA EXTRACTION

# loading training dataset ClassificationProblem1.txt using tab delimeter

dfTrain = pd.read_csv('ClassificationProblem1.txt', delimiter='\t')
dfTrain.head()
dfTrain.shape


# selecting categorical type column to transform into number representation 

cat_colmn = dfTrain.select_dtypes(include=[object])
cat_colmn.head()

# selecting non categorical column (Seperating cat and non_cat)
nonCat_colmn = dfTrain.select_dtypes(exclude=[object])
nonCat_colmn = nonCat_colmn.loc[:,'F1':'F22']
nonCat_colmn.head()
nonCat_colmn.shape
#-----------------------------------------------------------------
# Data preprocessing to transform catgorical type to number representation
# I used LableEncoder preprocessing sklear module

#label encoder
le = preprocessing.LabelEncoder()


lbl_encoded = cat_colmn.apply(le.fit_transform) 
lbl_encoded.head()

#-----------------------------------------------
# joining cat and non_cat column

prepro_frame = pd.concat([nonCat_colmn,lbl_encoded], axis=1)
prepro_frame.head()
#-----------------------------------------------------------
# VALIDATION STEP

#taking sample set from training file for validation

# taking 100000 for training and remaining for validation
X_tr = prepro_frame.iloc[:100000,:]
X_valid = prepro_frame.iloc[100000:,:]
X_tr.shape     #training matrix
X_valid.head()
X_valid.shape # Validation Matrix


# extracting Lable C "100000" for training and remaining for validation
y_tr = dfTrain['C'][:100000]

y_tr.head() #training Label Matrix

y_valid = dfTrain['C'][100000:]
y_valid.shape #validation Label Matrix

#----------------------------------------------------------------
# TRAINING CLASSIFIER

#training classifier model using two classifier
# clf = Random Forest classifier
# clfD = Decision Tree classifier

clf = RandomForestClassifier(n_jobs=5)
clfD = DecisionTreeClassifier()

clf.fit(X_tr,y_tr)  
clfD.fit(X_tr,y_tr)
#--------------------------------------------------------------
# VALIDATION STEP

# predicted label for X_valid "Sample from trainin data"
prdevalu = clf.predict(X_valid)  # predicted random forest
prdevaluD = clfD.predict(X_valid) # predicted decision tree

print(confusion_matrix(y_valid,prdevalu))
print(confusion_matrix(y_valid,prdevaluD))

print("Accuracy score for Random Forest classifier model",accuracy_score(y_valid,prdevalu))
print("Accuracy score for Decision Tree classifier model",accuracy_score(y_valid,prdevaluD))

print('recall_score for Random Forest classifier model',recall_score(y_valid,prdevalu))
print('recall_score for Decision Tree classifier model',recall_score(y_valid,prdevaluD))

#-------------------------------------------------------------
#random forest has less recall_score "less positive selection error rate of classifier"
#since problem statement is about X buys y hence False Positive should be optimized

#-------------------------------------------------------------
# RandomForestClassifier is choosen over DecisionTreeClassifier
#----------------------------------------------------------------
# TESTING PHASE

#extract training feature from test dataset

dfTest = pd.read_csv('Classification1Test.txt',delimiter='\t')
dfTest.head()
dfTest.shape

#selecting categ data from training set

cat_colmn_test = dfTest.select_dtypes(include=[object])
cat_colmn_test = cat_colmn_test.loc[:,'F15':'F16']
cat_colmn_test.head()

# selecting non categorical data from training set

nonCat_colmn_test = dfTest.select_dtypes(exclude=[object])
nonCat_colmn_test = nonCat_colmn_test.loc[:,'F1':'F22']
nonCat_colmn_test.head()
nonCat_colmn_test.shape
#---------------------------------------------------------------
# data preprocessing for test dataset

#label encoder for training dataset


lbl_encoded_test = cat_colmn_test.apply(le.fit_transform) 
lbl_encoded_test.head()
lbl_encoded_test.shape

#joining two dataframes cat and non_cat for traning set

X_test = pd.concat([nonCat_colmn_test,lbl_encoded_test],axis=1)
X_test.head()
X_test.shape

#-----------------------------------------------------------------
# PREDICTION STEP

predicted_values = clf.predict(X_test) #predicted label for test dataset

predicted_values.size

#-----------------------------------------------------------------
# WRITE OUTPUT TO FILE

# reading Index and writing to output.txt "Index <Tab delim> Class" 
ifile = open('Classification1Test.txt','r')
ofile = open('output11.txt','w', newline="")
reader = csv.reader(ifile, delimiter = '\t')
writer = csv.writer(ofile, delimiter = ' ')

writer.writerow(['Index', 'Class'])

indexList = []

for row in reader:
     indexList.append(row[0])

for i,j in zip(indexList[1:],predicted_values):
     writer.writerow([i, str(j)])



