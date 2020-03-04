
# coding: utf-8

# In[14]:


# Load libraries
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Set random seed
np.random.seed(0)

# Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Create a pipeline
pipe = Pipeline([('classifier', RandomForestClassifier())])

# Create space of candidate learning algorithms and their hyperparameters
search_space = [{'classifier': [LogisticRegression()],
                 'classifier__penalty': ['l1', 'l2'],
                 'classifier__C': np.logspace(0, 4, 10)},
                {'classifier': [RandomForestClassifier()],
                 'classifier__n_estimators': [10, 100, 1000],
                 'classifier__max_features': [1, 2, 3]},
               {'classifier': [SVC()],
               'classifier__kernel': ['rbf', 'linear'],
               'classifier__C': np.logspace(0, 4, 10)}]
                 
clf = GridSearchCV(pipe, search_space, cv=5, verbose=0)

best_model = clf.fit(X, y)

# View best model
print (best_model.best_estimator_.get_params()['classifier'])

# View cv scores for all trails
means = best_model.cv_results_['mean_test_score']
stds = best_model.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, best_model.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))


# In[11]:


print (best_model.best_estimator_.get_params()['classifier'])


# In[12]:


means = best_model.cv_results_['mean_test_score']
stds = best_model.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, best_model.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))


# In[7]:


from sklearn.svm import SVC


# In[9]:


SVC()

