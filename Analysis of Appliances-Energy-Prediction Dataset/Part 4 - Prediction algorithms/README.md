
### Part - 4 Prediction Algorithms

#### Libraries Used
sklearn.metrics
sklearn.cross_validation
sklearn.model_selection - used for Linear Regression
statsmodels.api - used for Multiple Linear Regression
sklearn.tree - used for DecisionTreeClassifier
sklearn.ensemble - used for RandomForestClassifier
xgboost - used for XGBoostClassifier
sklearn.ensemble.GradientBoostingClassifier
sklearn.preprocessing - used for calculating MinMaxScaler
sklearn.cluster.KMeans - used for k-means Clustering
sklearn.linear_model.LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix - to detect the truth value of our data
from sklearn.metrics import classification_report

###### Keras TensorFlow
keras.models import Sequential
keras.layers import LSTM
keras.layers import Dense
###### Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor  
from sklearn.cross_validation import LeaveOneLabelOut
from sklearn.grid_search import GridSearchCV

#### Added Dummy Variables-
timeofDay - Day and Night
activeStatus - awake and sleep
dayoftheweek - Monday to Sunday
WeekDayType - WeekDay and Weekend
season - spring and winter


#### User Defined Functions 
evaluate_model() - To calculate RMSE(Root Mean Squared Error), R^2, MAE(Mean Absolute Error) and MAPE(Mean Absolute Percentage Error)

#### Observations
We found out that Random Forest Regression gives better result wil R2 = 0.9 in training data and R2 = 0.6 in testing data


### For further Details please refer the Assignment2_Documentation.pdf file
