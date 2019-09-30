### Part 7 - Final Pipeline

### Libraries Used
1) Pandas
2) Numpy
3) Date
4) Logging
5) from sklearn.preprocessing import MinMaxScaler
6) from sklearn.ensemble import RandomForestRegressor - Best Model for our dataset
7) from sklearn.model_selection import train_test_split
8) from sklearn.pipeline import make_pipeline


### Dockerized 
Run the docker command : 

	1. docker built -f Dockerfile . -t \<image\>:tag
	2. docker run -it -p  \<image\>:tag
    
#### Pipelining

Logged - energydata_complete

#### Generated Columns
dayoftheweek
WeekDayType
activeStatus
NSM(Number of seconds from midnight)
weekOfTheYear
Season
Dummy

#### User Defined Functions
dayoftheweek(day) - Monday to Sunday
 weekdaytype(day)
 partOfTheDay(time)
 awakeTest(time)

#### Added Dummy Variables
1) timeofDay
2) activeStatus
3) dayoftheweek
4) weekDayType
5) season


#### Used MinMaxScaler
Fit the dataset using fit_transform
Split the data into training and testing dataset

#### RandomForestRegressor - Best Model
###### R-squared for Train: 0.94
###### R-squared for Test: 0.60




