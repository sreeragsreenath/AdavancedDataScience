# Luigi pipeline
get_ipython().system('pip install imblearn')
get_ipython().system('pip install pandas_profiling')
get_ipython().system('pip install ipywidgets ')
get_ipython().system('pip install luigi ')


# ## Import required libraries

# Data Collection and Transformations
import numpy as np
import pandas as pd
import datetime as dt
import time
import pickle
from sklearn.preprocessing import Imputer, StandardScaler
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
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor


# Plotting 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams['figure.figsize'] = [10,8]
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings


# ### Read the training and testing data


class Train_DataIngestion(luigi.Task):

    def run(self):
        trainingData= pd.read_csv("data/trainingData.csv")
    def output(self):
        return luigi.LocalTarget("/tmp/trainingData.csv")

class Test_DataIngestion(luigi.Task):

    def run(self):
        testingData = pd.read_csv("data/validationData.csv")
    def output(self):
        return luigi.LocalTarget("/tmp/validationData.csv")


# ### Drop unnecessary columns. 
# So here we need to predict the longitude and latitude of GPS, which can be done using the WAP columns. 
# Data Cleaning and processing

class DataPreProcessing(luigi.Task):

    def requires(self):
        return Train_DataIngestion()

    def run(self):
        fb = FeatureBuilder(pandas.read_csv(Train_DataIngestion().output().path))
        trainingData = fb.featurize()
        print "In Data Pre Processing"
        X_train = trainingData.drop(['FLOOR', 'BUILDINGID','SPACEID','combine','LONGITUDE','LATITUDE','RELATIVEPOSITION','USERID','PHONEID','TIMESTAMP'], axis=1)
		y_train = trainingData[['LONGITUDE','LATITUDE']]
		X_train = X_train.values
		y_train = y_train.values
        X_train.to_csv(self.output().path, index=False)
        y_train.to_csv(self.output().path, index=False)

    def output(self):
    	 return { 'output1' : luigi.LocalTarget("/tmp/X_train.csv"),
                  'output2' : luigi.LocalTarget("/tmp/y_train.csv") }

#normal method
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

X_train = trainingData.drop(['FLOOR', 'BUILDINGID','SPACEID','combine','LONGITUDE','LATITUDE','RELATIVEPOSITION','USERID','PHONEID','TIMESTAMP'], axis=1)
y_train = trainingData[['LONGITUDE','LATITUDE']]
X_train = X_train.values
y_train = y_train.values
rmse_dict = {} 

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

#Implementing random forest
class RandomForestRegressor(luigi.Task):
    def run(self):
    	print("RandomForestRegressor")
        classifier = RandomForestRegressor(max_features=10 , n_jobs=-1 )
        self.input()['input11']['output1'].open("r") as infile1,  
                 self.input()['input11']['output2'].open("r") as infile2,
        fb = FeatureBuilder(pandas.read_csv(['input11']['output1'].output().path))
        X_train = fb.featurize()
        fb1 = FeatureBuilder(pandas.read_csv(['input11']['output2'].output().path))
        y_train = fb1.featurize()
		classifier.fit(X_train, y_train)
		RandomForestRegressorModel=evaluate_model(classifier, "RandomForestRegressor",classifier,X_train,y_train, X_test , y_test)
		pickle.dump(RandomForestRegressorModel,f )
		request = Request('http://127.0.0.1:5000/loadModels/')
        try:
            print("Reloading models")
            response = urlopen(request)
        except URLError, e:
            "No model", e
    def output(self):
        return luigi.LocalTarget("/tmp/RandomForestRegressor.pkl")


if __name__ == '__main__':
    luigi.run()

class Train(luigi.Task):

    def requires(self):
        return DataPreProcessing()

    def run(self):
        sales_model = train_model_ridge(pandas.read_csv(DataPreProcessing().output().path))
        with open(self.output().path, 'wb') as f:
            pickle.dump(sales_model,f )
        request = Request('http://127.0.0.1:5000/loadModels/')
        try:
            print("Reloading models")
            response = urlopen(request)
        except URLError as e:
            "No Roseman Sales Prediction API", e
