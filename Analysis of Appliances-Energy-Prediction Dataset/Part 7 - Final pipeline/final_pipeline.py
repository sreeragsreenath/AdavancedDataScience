import pandas as pd
import time
import numpy as np
import datetime
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

logfilename = 'log_pipeLine.txt'
print('LOG File name : log_pipeLine.txt')
logging.basicConfig(filename=logfilename, level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug('Program Started')
print('Program Started')

logging.debug('Loading Data into Dataframe')
print('Loading Data into Dataframe')
try :  
    data= pd.read_csv("energydata_complete.csv")
    logging.debug('Data Size'+str(data.shape) )
    
except :
    logging.ERROR('Data logging failed')



logging.debug("Tranforming date time")
print("Tranforming data columns")
data["date_time"] = pd.to_datetime(data["date"],format="%Y-%m-%d %H:%M:%S")


def dayoftheweek(day):
    if(day==0):
        return("Monday")
    if(day==1):
        return("Tuesday")
    if(day==2):
        return("Wednesday")
    if(day==3):
        return("Thurday")
    if(day==4):
        return("Friday")
    if(day==5):
        return("Saturday")
    if(day==6):
        return("Sunday")

logging.debug('Creating column Day of the week') 
data["dayoftheweek"] = data['date_time']
data["dayoftheweek"] = data['dayoftheweek'].apply(lambda x: dayoftheweek(x.dayofweek))
data.groupby('dayoftheweek').count()["date_time"]


logging.debug('Creating column weekdaytype') 
def weekdaytype(day):
    if(day=="Saturday" or day == "Sunday"):
        return "weekend"
    else:
        return "Weekday"

data["WeekDayType"] = data["dayoftheweek"]
data["WeekDayType"] = data['WeekDayType'].apply(lambda x: weekdaytype(x))
data.groupby('WeekDayType').count()["date_time"]



def partOfTheDay(time):
    day1 = pd.to_datetime('18:00:00',format="%H:%M:%S")
    day2 = pd.to_datetime('6:00:00',format="%H:%M:%S")
    if(time<day1.time() and time >= day2.time()):
        return "Day"
    else:
        return "Night"


data['timeofDay'] = data["date_time"].map(lambda x: partOfTheDay(x.time()))


data.groupby('timeofDay').count()["date_time"]



def awakeTest(time):
    day1 = pd.to_datetime('8:00:00',format="%H:%M:%S")
    day2 = pd.to_datetime('22:00:00',format="%H:%M:%S")
    if(time>=day1.time() and time < day2.time()):
        return "awake"
    else:
        return "sleep"


logging.debug('Creating column activeStatus') 
data['activeStatus'] = data["date_time"].map(lambda x: awakeTest(x.time()))



data.groupby('activeStatus').count()["date_time"]



logging.debug('Creating column Number of seconds to midnight') 
data['NSM'] = pd.to_datetime(data['date_time'])
data['NSM']  = (data['NSM'].dt.hour*60 + data['NSM'].dt.minute)*60 + data['NSM'].dt.second

data["weekOfTheYear"] = data['date_time'].apply(lambda x: x.isocalendar()[1])



from datetime import date, datetime
Y = 2000 # dummy leap year to allow input X-02-29 (leap day)
seasons = [('winter', (date(Y,  1,  1),  date(Y,  3, 20))),
           ('spring', (date(Y,  3, 21),  date(Y,  6, 20))),
           ('summer', (date(Y,  6, 21),  date(Y,  9, 22))),
           ('autumn', (date(Y,  9, 23),  date(Y, 12, 20))),
           ('winter', (date(Y, 12, 21),  date(Y, 12, 31)))]
def get_season(now):
    if isinstance(now, datetime):
        now = now.date()
    now = now.replace(year=Y)
    return next(season for season, (start, end) in seasons
                if start <= now <= end)


# In[18]:


logging.debug('Creating column Season') 
data['season'] = data['date_time'].apply(lambda x: get_season(x))


# In[19]:


logging.debug('Creating Dummy Column') 
data = pd.get_dummies(data, columns=["timeofDay","activeStatus",'dayoftheweek','WeekDayType','season'])


# In[20]:


#Select response y and Training set X
k = list(data)
k
my_cols = [
 'lights',
 'T1',
 'RH_1',
 'T2',
 'RH_2',
 'T3',
 'RH_3',
 'T4',
 'RH_4',
 'T5',
 'RH_5',
 'T6',
 'RH_6',
 'T7',
 'RH_7',
 'T8',
 'RH_8',
 'T9',
 'RH_9',
 'T_out',
 'Press_mm_hg',
 'RH_out',
 'Windspeed',
 'Visibility',
 'Tdewpoint',
 'NSM',
 'weekOfTheYear',
 'timeofDay_Day',
 'timeofDay_Night',
 'activeStatus_awake',
 'activeStatus_sleep',
 'dayoftheweek_Friday',
 'dayoftheweek_Monday',
 'dayoftheweek_Saturday',
 'dayoftheweek_Sunday',
 'dayoftheweek_Thurday',
 'dayoftheweek_Tuesday',
 'dayoftheweek_Wednesday',
 'WeekDayType_Weekday',
 'WeekDayType_weekend',
 'season_spring',
 'season_winter']
X = data[my_cols]
y = data['Appliances']



print("using Random forest regression model")
min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, test_size = 0.25)

exported_pipeline = Pipeline([
  ('feature_selection', SelectFromModel(ExtraTreesClassifier())),
  ('scaling', MinMaxScaler()),
  ('classification', RandomForestRegressor(n_estimators = 300, max_features=4)),

])

exported_pipeline.fit(X_train, y_train)


print ("R-squared for Train: %.2f" %exported_pipeline.score(X_train, y_train))
print ("R-squared for Test: %.2f" %exported_pipeline.score(X_test, y_test))

logging.debug("R-squared for Train: %.2f" %exported_pipeline.score(X_train, y_train))
logging.debug(("R-squared for Test: %.2f" %exported_pipeline.score(X_test, y_test)))

logging.debug("Process complete")
print("Process complete")

