# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:12:38 2019

@author: Mohamed Zeitoun
"""

# In[]:

# load basic libraries 
import nltk
import numpy as np
import pandas as pd 
import glob
import warnings
warnings.filterwarnings('ignore')


# In[]:

# Load train dataset into dataframe

tweet_files = glob.glob("./twitter-201?train.txt")

# Load train dataset into dataframe

li = []

for filename in tweet_files:
    df = pd.read_csv(filename, index_col=None, names=['Timestamp', 'Sentiment', 'Tweet'], sep='\t')
    li.append(df)

tweets = pd.concat(li, axis=0, ignore_index=True)

from sklearn.preprocessing import LabelEncoder

# Label encoding test & training labels
le = LabelEncoder()
Y_train = le.fit_transform(tweets['Sentiment'])

X_train = tweets['Tweet']

# In[]:

#List of different models
model = [
#            "CountVectorizer + Naïve Bayes Multinomial", 
#            "TFIDFVectorizer + Naïve Bayes Multinomial", 
#            "CountVectorizer with uni-grams and bi-grams + Naïve Bayes Multinomial", 
#            "CountVectorizer + Logistic Regression", 
#            "TFIDFVectorizer + Logistic Regression", 
#            "CountVectorizer with uni-grams and bi-grams + Logistic Regression",
#            "TFIDFVectorizer + SVM (Linear Kernel)",
#            "CountVectorizer + SVM (Linear Kernel)",
#            "CountVectorizer with uni-grams and bi-grams + SVM (Linear Kernel)",
#            "TFIDFVectorizer + SVM (RBF)",
#            "CountVectorizer + SVM (RBF)",
#            "CountVectorizer with uni-grams and bi-grams + SVM (RBF)",
#            "CountVectorizer + Random Forest",
#            "Majority Voting"
#            "boosting with majorty voting"
            "bagging with logestic regression"
            
        ]

#intialize the output matrix
result = pd.DataFrame(columns=['Accuracy', 'FScore'])

#Load libraries needed for classification 
from sklearn.pipeline import Pipeline
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier,GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
 

#define the 10-fold
kfold = KFold(n_splits=10, shuffle=True, random_state=1234)

# In[]:

#loop on each model
for i in model:
    if i == 'CountVectorizer + Naïve Bayes Multinomial':
        #CountVectorizer + Naïve Bayes Multinomial pipeline
        pipeline = Pipeline([
        ('CountVectprizer', CountVectorizer()),
        ('naive_bayes_Multinomial', naive_bayes.MultinomialNB())
        ])
    elif i == 'TFIDFVectorizer + Naïve Bayes Multinomial':
        #TFIDFVectorizer + Naïve Bayes Multinomial pipeline
        pipeline = Pipeline([
        ('TFIDFVectprizer', TfidfVectorizer()),
        ('naive_bayes_Multinomial', naive_bayes.MultinomialNB())
        ])
    elif i == 'CountVectorizer with uni-grams and bi-grams + Naïve Bayes Multinomial':
        #CountVectorizer with uni-grams and bi-grams + Naïve Bayes Multinomial pipeline
        pipeline = Pipeline([
        ('CountVectorizer', CountVectorizer(ngram_range=(1,2))),
        ('naive_bayes_Multinomial', naive_bayes.MultinomialNB())
        ])
    elif i == 'CountVectorizer + Logistic Regression':
        #CountVectorizer + Logistic Regression pipeline
        pipeline = Pipeline([
        ('CountVectorizer', CountVectorizer()),
        ('LogisticRegression', LogisticRegression())
        ])
    elif i == 'TFIDFVectorizer + Logistic Regression':
        #TFIDFVectorizer + Logistic Regression pipeline
        pipeline = Pipeline([
        ('TFIDFVectorizer', TfidfVectorizer()),
        ('LogisticRegression', LogisticRegression())
        ])
    elif i == 'CountVectorizer with uni-grams and bi-grams + Logistic Regression':
        #CountVectorizer with uni-grams and bi-grams + Logistic Regression pipeline
        pipeline = Pipeline([
        ('CountVectorizer', CountVectorizer(ngram_range=(1,2))),
        ('LogisticRegression', LogisticRegression())
        ])
    elif i == 'CountVectorizer + SVM (Linear Kernel)':
        #CountVectorizer + SVM (Linear Kernel) pipeline
        pipeline = Pipeline([
        ('CountVectorizer', CountVectorizer()),
        ('SVM_linear_kernel', svm.LinearSVC())
        ])
    elif i == 'TFIDFVectorizer + SVM (Linear Kernel)':
        #TFIDFVectorizer + SVM (Linear Kernel) pipeline
        pipeline = Pipeline([
        ('TFIDFVectorizer', TfidfVectorizer()),
        ('SVM_linear_kernel', svm.LinearSVC())
        ])
    elif i == 'CountVectorizer with uni-grams and bi-grams + SVM (Linear Kernel)':
        #CountVectorizer with uni-grams and bi-grams + SVM (Linear Kernel) pipeline
        pipeline = Pipeline([
        ('CountVectorizer', CountVectorizer(ngram_range=(1,2))),
        ('SVM_linear_kernel', svm.LinearSVC())
        ])
    elif i == 'CountVectorizer + SVM (RBF)':
        #CountVectorizer + SVM (RBF) pipeline
        pipeline = Pipeline([
        ('CountVectorizer', CountVectorizer()),
        ('SVM_RBF', svm.SVR(kernel='rbf'))
        ])
    elif i == 'TFIDFVectorizer + SVM (RBF)':
        #TFIDFVectorizer + SVM (RBF) pipeline
        pipeline = Pipeline([
        ('TFIDFVectorizer', TfidfVectorizer()),
        ('SVM_RBF', svm.SVR(kernel='rbf'))
        ])
    elif i == 'CountVectorizer with uni-grams and bi-grams + SVM (RBF)':
        #CountVectorizer with uni-grams and bi-grams + SVM (RBF) pipeline
        pipeline = Pipeline([
        ('CountVectorizer', CountVectorizer(ngram_range=(1,2))),
        ('SVM_RBF', svm.SVR(kernel='rbf'))
        ])
    elif i == 'CountVectorizer + Random Forest':
        #CountVectorizer + Random Forest pipeline
        pipeline = Pipeline([
        ('CountVectorizer', CountVectorizer()),
        ('RandomForest', RandomForestClassifier())
        ])
    elif i == 'Majority Voting':
        #Majority Voting Ensamble
        LRpipeline = Pipeline([
        ('CountVectorizer', CountVectorizer(ngram_range=(1,2))),
        ('LogisticRegression', LogisticRegression())
        ])
    
        SVMpipeline = Pipeline([
        ('CountVectorizer', CountVectorizer(ngram_range=(1,2))),
        ('SVM_linear_kernel', svm.LinearSVC())
        ])
    
        NBpipeline = Pipeline([
        ('CountVectprizer', CountVectorizer()),
        ('naive_bayes_Multinomial', naive_bayes.MultinomialNB())
        ])
    
        RFpipeline = Pipeline([
        ('CountVectorizer', CountVectorizer()),
        ('RandomForest', RandomForestClassifier())
        ])
#        pipeline = VotingClassifier( estimators= [('lr',LRpipeline),('svm',SVMpipeline),('nb',NBpipeline),('rf',RFpipeline)], voting = 'hard')
        pipeline = VotingClassifier( estimators= [('lr',LRpipeline),('svm',SVMpipeline)], voting = 'hard')
    elif i == 'boosting with majorty voting':
        #Majority Voting Ensamble
        ada_boospipeline = Pipeline([
        ('CountVectorizer', CountVectorizer(ngram_range=(1,2))),        
        ('ada_boost', AdaBoostClassifier())
        ])
    
        grad_boostpipeline = Pipeline([
        ('CountVectorizer', CountVectorizer(ngram_range=(1,2))),        
        ('grad_boost', GradientBoostingClassifier())
        ])
#    
#        xgb_boostpipeline = Pipeline([
#       ('xgb_boost', XGBClassifier())
#        ])
#        pipeline = VotingClassifier( estimators= [('lr',LRpipeline),('svm',SVMpipeline),('nb',NBpipeline),('rf',RFpipeline)], voting = 'hard')
        #pipeline = VotingClassifier( estimators= [('ada_boost',ada_boospipeline),('grad_boost',grad_boostpipeline),('xgb_boost',xgb_boostpipeline)], voting = 'hard')
        pipeline = VotingClassifier( estimators= [('ada_boost',ada_boospipeline),('grad_boost',grad_boostpipeline)], voting = 'hard')
    elif i == 'bagging with logestic regression':
        #Majority Voting Ensamble
        pipeline = Pipeline([
        ('CountVectorizer', CountVectorizer(ngram_range=(1,2))),        
        ('bagging', BaggingClassifier(base_estimator=LogisticRegression(), n_estimators=5, random_state=1))
        ])
         
    else:
        print("ERROR: Model Not Supported!")
           
    #intialize the mean absolute error counter
    accuracy = 0.0
    Fscore = 0.0
    
    #K-fold cross validation
    for fold, (train_index, val_index) in enumerate(kfold.split(X_train, Y_train)):
        train_x, train_y = X_train.iloc[train_index], Y_train[train_index]
        val_x, val_y = X_train.iloc[val_index], Y_train[val_index]
        
        #Model fit & Prediction
        pipeline.fit(train_x, train_y)
        predictions = pipeline.predict(val_x)
        
        #Calculate the Accuracy & F-Score 
        accuracy += metrics.accuracy_score(val_y, predictions.round())
        Fscore += metrics.f1_score(val_y, predictions.round(), average='macro')
        
    accuracy /= kfold.get_n_splits()
    Fscore /= kfold.get_n_splits()
    
    print(i + ":")
    print("Accuracy = {}".format(accuracy.round(2)))
    print("F-Score = {}".format(Fscore.round(2)))
    
#    scores = cross_val_score(pipeline, X_train, Y_train, cv=10 )
#    fscores = cross_val_score(pipeline, X_train, Y_train, cv=10, scoring='f1_macro')
#    print(i + ":")
#    print("Accuracy = {}".format(scores.mean()))
#    print("F-Score = {}".format(fscores.mean()))
    
    result.loc[i,'Accuracy']=accuracy.round(2)
    result.loc[i,'FScore']=Fscore.round(2)
        
display(result.sort_values(by='FScore', ascending=False))

#votingC = VotingClassifier( estimators= [('lr',clf1),('dt',clf2),('nb',clf3)], voting = 'hard')
#votingC.fit(train_x, train_y)
#predicted = votingC.predict(sms_test_vectorized)
#acc = metrics.accuracy_score(test_labels,predicted)
#print ('accuracy =  %0.2f' %(acc*100)+'%')


#param_grid = {
# 'bootstrap': [True, False],
# 'bootstrap_features': [True, False],    
# 'n_estimators': [5, 10, 15],
# 'max_samples' : [0.6, 0.8, 1.0],
# 'base_estimator__bootstrap': [True, False],    
# 'base_estimator__n_estimators': [100, 200, 300],
# 'base_estimator__max_features' : [0.6, 0.8, 1.0]
#}
#
#grid_search=GridSearchCV(BaggingClassifier(base_estimator=RandomForestClassifier()), param_grid=param_grid, cv=10)
#grid_search=GridSearchCV(AdaBoostClassifier(param_grid=param_grid, cv=10)








