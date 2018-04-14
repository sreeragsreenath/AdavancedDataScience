
Exploratory Data Analysis 
Conduct EDA using Python libraries - 
Seaborn, Numpy, MissingNo, Pickle, Plotly, Pandas, Scipy, Sklearn, Matplotlib
Dropped unnecessary columns. 
  Predicted the longitude and latitude of GPS by using the WAP columns. 

Observations :  In our training samples, building 2 has the clear majority with it's count being slightly lower than the sum  of building 0 and building 1. Building 0 and building 1 have roughly the same representation in the training data.
Buildings 0 and 1 have 4 floors whereas Building 2 has 5 floors.
Expectedly, the samples from Building 2 are consistently the highest across all the floors.

Models Used-
Random Forest Regressor
Extra Trees Regressor
Random Forest Classifier
Extra Trees Classifier

Random Forest Regressor - gives 99%  r-squared for training data and 97%  r-squared for testing Data
Extra Trees Regressor - gives 99% r-squared for training Data and RMSE value is 5.09, it gives 98% for testing Data and RMSE value is  11.9
Random Forest Classifier - gives 99%  r-squared for training data and 86%  r-squared for testing Data
Extra Trees Classifier - gives 99%  r-squared for training data and 88% r-squared for testing Data

