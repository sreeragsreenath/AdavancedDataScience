
# coding: utf-8

# In[1]:


get_ipython().system('pip install imblearn')
get_ipython().system('pip install pandas_profiling')
get_ipython().system('pip install ipywidgets ')
get_ipython().system('pip install luigi ')


# ## Import required libraries

# In[18]:


# Data Collection and Transformations
import numpy as np
import pandas as pd
import datetime as dt
import time
import pickle
from sklearn.preprocessing import Imputer, StandardScaler

# Statistical Testing
# import statsmodels.api as sm
# from statsmodels.formula.api import ols
import scipy

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve, validation_curve

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
# Class imbalance 
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

# Plotting 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams['figure.figsize'] = [10,8]
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings


# ### Read the training and testing data

# In[19]:


trainingData= pd.read_csv("data/trainingData.csv")
trainingData.head()


# In[20]:


testingData = pd.read_csv("data/validationData.csv")
testingData.head()


# In[21]:


# testingData["combine"] = testingData["FLOOR"].map(str) + testingData["BUILDINGID"].map(str) + testingData["SPACEID"].map(str)


# In[22]:


# trainingData["combine"] = trainingData["FLOOR"].map(str) + trainingData["BUILDINGID"].map(str) + trainingData["SPACEID"].map(str)


# ### Drop unnecessary columns. 
# So here we need to predict the longitude and latitude of GPS, which can be done using the WAP columns. 

# In[23]:


X_train = trainingData.drop(['FLOOR', 'BUILDINGID','SPACEID','combine','LONGITUDE','LATITUDE','RELATIVEPOSITION','USERID','PHONEID','TIMESTAMP'], axis=1)
y_train = trainingData[['LONGITUDE','LATITUDE']]


# Convert it into an array, removing the headers.

# In[24]:


X_train = X_train.values
y_train = y_train.values


# In[11]:


y_train


# In[25]:


X_test = testingData.drop(['FLOOR', 'BUILDINGID','SPACEID','combine','LONGITUDE','LATITUDE','RELATIVEPOSITION','USERID','PHONEID','TIMESTAMP'], axis=1)
y_test = testingData[['LONGITUDE','LATITUDE']]


# In[26]:


# X_test = (X_train.replace(to_replace=100,value=np.nan))

# # Perform the same transform on Test data
# y_test = (y_test.replace(to_replace=100,value=np.nan))


# In[27]:



# print("Before sample removal:", len(X_train))

# y_train = (y_train.loc[X_train.notnull().any(axis=1),:])

# X_train = (X_train.loc[X_train.notnull().any(axis=1),:])


# print("After sample removal:", len(X_train))


# In[29]:



X_exp_train = np.power(10,X_train/10,)
X_exp_test = np.power(10,X_test/10)


# In[30]:


X_exp_train


# In[ ]:


X = testingData.drop(['FLOOR', 'BUILDINGID','SPACEID','combine'], axis=1)
y = testingData['combine']


# In[31]:


X = trainingData.drop(['FLOOR', 'BUILDINGID','SPACEID','combine'], axis=1)
y = trainingData['combine']


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# In[33]:


#Importing Libraries
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.cross_validation import cross_val_score


rmse_dict = {}    
def rmse(correct,estimated):
    rmse_val = np.sqrt(mean_squared_error(correct,estimated)) 
    return rmse_val

# Generating the Table Frame for metrics
evluation_table = pd.DataFrame({  'Model_desc':[],
                        'Model_param':[],
                        'r2_train': [],
                        'r2_test': [],
                        'rms_train':[], 
                        'rms_test': [],
                        'mae_train': [],
                        'mae_test': [],
                        'mape_train':[],
                        'mape_test':[],
                        'cross_val_score' : []})


# Evaluating the model
def evaluate_model(model, model_desc,model_param, X_train, y_train, X_test, y_test):
    global evluation_table
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
        
    
    try:
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
    except:
        r2_train = "not calculated"
        r2_test = "not calculated"
    try:
        rms_train = rmse(y_train, y_train_pred)
        rms_test = rmse(y_test, y_test_pred)
    except:
        rms_train = "not calculated"
        rms_test = "not calculated"
    try:
        mae_train = mean_absolute_error(y_train, y_train_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
    except:
        mae_train = "not calculated"
        mae_test = "not calculated"
    try:
        mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
        mape_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    except:
        mape_train = "not calculated"
        mape_test = "not calculated"
    
    
    try:
        cv_score = cross_val_score(model, X_train, y_train, cv=10)
        cv_score = cv_score.mean()
    except:
        cv_score = "Not calulated"
        
    model_param = pd.DataFrame({'Model_desc':[model_desc],
                            'Model_param':[model_param],
                            'r2_train': [r2_train],
                            'r2_test': [r2_test],
                            'rms_train':[rms_train], 
                            'rms_test': [rms_test],
                            'mae_train': [mae_train],
                            'mae_test': [mae_test],
                            'mape_train':[mape_train],
                            'mape_test':[mape_test],
                            'cross_val_score' : [cv_score]})

    evluation_table = evluation_table.append([model_param])
 
    return evluation_table


# In[34]:


from sklearn.ensemble import RandomForestRegressor


# In[35]:


classifier = RandomForestRegressor(max_features=10 , n_jobs=-1 )
classifier.fit(X_train, y_train)


# In[36]:


evaluate_model(classifier, "RandomForestRegressor",classifier,X_train,y_train, X_test , y_test)


# In[37]:


model_param_grid = {}
model_scores = {}


# In[38]:


def nested_crossval(reg_list,reg_labels, model_param_grid=model_param_grid, model_scores = model_scores,
                    X = X_train, y= y_train, label_extension = None):
    '''
    Inputs:
    reg_model        : List of Regression model instances
    reg_label        : List of Regression model labels
    model_param_grid : List of parameter grids
    X                : explanatory variables 
    y                : response variable array
    model_scores     : Dictionary to store nested cross-validation scores
    label_extension  : Extension to regression label in model_scores key
    
    Outputs:
    model_scores     : Updated dictionary of nested cross-validation scores
    '''

    
    for reg_model, reg_label in zip(reg_list, reg_labels):
    
        #print(param_grid)
    
        gs = (GridSearchCV(estimator=reg_model, 
                            param_grid=model_param_grid[reg_label], 
                            cv=2,
                            scoring = 'neg_mean_squared_error',
                            n_jobs = 1))
    
        scores = cross_val_score(estimator=gs,
                                 X=X,
                                 y=y,
                                 cv=5,
                                 scoring='neg_mean_squared_error')
        scores = np.sqrt(np.abs(scores))
        
        if label_extension:
            reg_label += '_' + label_extension
        
        print("RMSE: %0.2f (+/- %0.2f) [%s]"
              % (scores.mean(), scores.std(), reg_label))
        
        model_scores[reg_label] = scores
        
    
    return model_scores


# In[ ]:



## Linear Models

# Ridge Regression
pipe_ridge = Pipeline([('scl', StandardScaler()),
            ('reg', Ridge(random_state=1))])

# Lasso
pipe_lasso = Pipeline([('scl', StandardScaler()),
            ('reg', Lasso(random_state=1))])

param_grid_lm= {
    'reg__alpha':[0.01,0.1,1,10],
}

reg_lm = [pipe_ridge,pipe_lasso]
reg_labels_lm = ['Ridge','Lasso']
model_param_grid['Ridge'] = param_grid_lm
model_param_grid['Lasso'] = param_grid_lm

model_scores = nested_crossval(reg_lm,reg_labels_lm)


# In[ ]:



# Random Forests
reg_rf = RandomForestRegressor(random_state=1)

# Extra Trees
reg_et = ExtraTreesRegressor(random_state=1)

param_grid_tree = {
    'n_jobs': [-1],
    'n_estimators': [10,30,50,70,100],
    'max_features': [0.25,0.5,0.75],
    'max_depth': [3,6,9,12],
    'min_samples_leaf': [5,10,20,30]
}

reg_tree = [reg_rf,reg_et]
reg_labels_tree = ['Random Forests','Extra Trees']
model_param_grid['Random Forests'] = param_grid_tree
model_param_grid['Extra Trees'] = param_grid_tree

model_scores = nested_crossval(reg_tree,reg_labels_tree)


# In[ ]:


pipe_knn = Pipeline([('scl', StandardScaler()),
            ('reg', KNeighborsRegressor())])

grid_param_knn = {
    'reg__n_neighbors': [2,3,5,7],
    'reg__weights': ['uniform','distance'],
    'reg__metric': ['euclidean','minkowski','manhattan'],
    'reg__n_jobs': [-1]
}

model_param_grid['KNN'] = grid_param_knn

model_scores = nested_crossval([pipe_knn],['KNN'])


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(X_train, y_train)


# In[ ]:


evaluate_model(neigh, "RandomForestClassifier",neigh,X_train,y_train, X_test , y_test)


# In[ ]:


from tpot import TPOTRegressor
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# In[14]:


regressor_config_dict = {
        'sklearn.ensemble.ExtraTreesRegressor': {
        'n_estimators': [100],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.GradientBoostingRegressor': {
        'n_estimators': [100],
        'loss': ["ls", "lad", "huber", "quantile"],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05),
        'alpha': [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    },

    'sklearn.ensemble.AdaBoostRegressor': {
        'n_estimators': [100],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'loss': ["linear", "square", "exponential"],
        'max_depth': range(1, 11)
    },

    'sklearn.tree.DecisionTreeRegressor': {
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },

    'sklearn.neighbors.KNeighborsRegressor': {
        'n_neighbors': range(1, 101),
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    },

    'sklearn.linear_model.LassoLarsCV': {
        'normalize': [True, False]
    },

    'sklearn.svm.LinearSVR': {
        'loss': ["epsilon_insensitive", "squared_epsilon_insensitive"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'epsilon': [1e-4, 1e-3, 1e-2, 1e-1, 1.]
    },

    'sklearn.ensemble.RandomForestRegressor': {
        'n_estimators': [100],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },
    'sklearn.linear_model.RidgeCV': {
    },

    'xgboost.XGBRegressor': {
        'n_estimators': [100],
        'max_depth': range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'nthread': [1]
    },
    'sklearn.preprocessing.Binarizer': {
        'threshold': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.decomposition.FastICA': {
        'tol': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.cluster.FeatureAgglomeration': {
        'linkage': ['ward', 'complete', 'average'],
        'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    },

    'sklearn.preprocessing.MaxAbsScaler': {
    },

    'sklearn.preprocessing.MinMaxScaler': {
    },

    'sklearn.preprocessing.Normalizer': {
        'norm': ['l1', 'l2', 'max']
    },

    'sklearn.kernel_approximation.Nystroem': {
        'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
        'gamma': np.arange(0.0, 1.01, 0.05),
        'n_components': range(1, 11)
    },

    'sklearn.decomposition.PCA': {
        'svd_solver': ['randomized'],
        'iterated_power': range(1, 11)
    },

    'sklearn.preprocessing.PolynomialFeatures': {
        'degree': [2],
        'include_bias': [False],
        'interaction_only': [False]
    },

    'sklearn.kernel_approximation.RBFSampler': {
        'gamma': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.preprocessing.RobustScaler': {
    },

    'sklearn.preprocessing.StandardScaler': {
    },

    'tpot.builtins.ZeroCount': {
    },

    'tpot.builtins.OneHotEncoder': {
        'minimum_fraction': [0.05, 0.1, 0.15, 0.2, 0.25],
        'sparse': [False]
    },

    # Selectors
    'sklearn.feature_selection.SelectFwe': {
        'alpha': np.arange(0, 0.05, 0.001),
        'score_func': {
            'sklearn.feature_selection.f_regression': None
        }
    },

    'sklearn.feature_selection.SelectPercentile': {
        'percentile': range(1, 100),
        'score_func': {
            'sklearn.feature_selection.f_regression': None
        }
    },

    'sklearn.feature_selection.VarianceThreshold': {
        'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    },

    'sklearn.feature_selection.SelectFromModel': {
        'threshold': np.arange(0, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesRegressor': {
                'n_estimators': [100],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    }

}


# In[15]:


tpot = TPOTRegressor(generations=10, population_size=10, verbosity=2, n_jobs=-1, periodic_checkpoint_folder="./optCode/")
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_final_pipeline.py')


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy


# In[ ]:


# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, test_size = 0.25)

# define random forest classifier, with utilising all cores and
# sampling in proportion to y labels
forest = RandomForestClassifier(n_jobs=-1)
 
# define Boruta feature selection method
feat_selector = BorutaPy(forest, n_estimators='auto', verbose=2)
 
# find all relevant features
feat_selector.fit(X_train, y_train)
 
# check selected features
feat_selector.support_
 
# check ranking of features
feat_selector.ranking_


# In[ ]:



from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# In[ ]:


min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)
# split into train and test sets
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[ ]:


train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# In[ ]:


# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=1, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# In[ ]:


yhat = model.predict(test_X)
yhat
rmse = sqrt(mean_squared_error(test_y, yhat))
print('Test RMSE: %.3f' % rmse)
r2_score(test_y, yhat)

