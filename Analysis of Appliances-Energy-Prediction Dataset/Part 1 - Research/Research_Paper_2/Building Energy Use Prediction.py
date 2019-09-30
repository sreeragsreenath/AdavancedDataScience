
# coding: utf-8

# # Research Topic:  AI Based Energy based building energy use prediction
#  
# 

# ## Summary :-

# The author discussed about how to predict the consumption of energy used by a building. This prediction includes various AI-based approaches like multiple linear regression, Artificial Neural Network, Support vector regression and Ensemble methods. The AI based approach largely depends on the past data for forecasting the energy.

# ## Introduction :-

# Since 1990s, researchers developed various simulation tools for predicting the energy consumption. These tools can be divided as-  engineering method, AI-based method, and hybrid method. The engineering method and hybrid method both used white box, but later they were not feasible so they carried their research using AI method, which uses black box. It predicts depending upon the correlated variables such as environmental conditions, building characteristics, and occupancy status.  

# ## Current trends :-

# The classification of the current trends depends on the following factors-
# Building type – are classified into commercial, residential, educational and research, and other building types. From the pie chart, it’s observed that Education and Research consist 42% and Commercial energy consist 32% but this data is not complete due to the unavailability of the household data. As this data doesn’t include household data, we can’t take this into consideration.
# 

# ## Energy Type :–

# The author distributed energy type into 5 categories but in my view, some categories was not required like Heating & Cooling as it already described its separate distribution. Also, the Building energy includes the overall distribution of the energy.

# ## Prediction Time Scale :-

# Most of the researchers chose hourly prediction rather than daily or yearly basis. From my point of view it should mention monthly prediction as we saw energy type, so according to the difference in temperature we can predict in which month the energy consumption is more or less.

# ## Input Data Type :-

# This data includes Meteorology, Occupancy and Others. From most of the researchers, it’s observation tells that most of the data depends on Meteorology but it should depend on Occupancy as this factor have significant impacts on building energy use and enhanced the performance of Prediction. But acquiring the data for occupancy is difficult. In my view this type is just an extra field for our prediction.  

# ## Prediction Methods :-

# The author proposed two methods - Single Prediction and Ensemble Method.
# 
# Single Prediction method - It uses one variable for predicting the energy use, it includes various methods like MLR, ANN and SVR. Each of these methods depends on particular factors like in MLR, it works well under linear problems. For ANN, many studies observed that it performs better than regression for short-term forecasting, as it detects complex non-linear relationship between inputs and outputs. For SVR, it gives high level of prediction accuracy compared to the other two methods but it requires appropriate parameters. From my perspective, I think that we can’t say that which method is best for predicting rather that I would suggest that all the methods have their own significance, it depends on the available data source and the requirements, you should select the model
# 
# After observing the flaws into Single prediction method, Researchers proposed Ensemble method to overcome the flaws. Ensemble model uses a framework for prediction, which includes data collection, data preprocessing, learning algorithm, Base Learner and at last prediction. This model is also classified into two, based on the requirements like In homogeneous ensemble method there is only one Learning Algorithm for different training datasets, this is appropriate for unstable learning algorithms. Another model is Heterogeneous Ensemble model which uses different Learning Algorithms for a particular dataset. This is used to solve regression as well as classification problems.    
# 

# ## Conclusion :-

# The author discussed many approaches to predict the energy consumption but each method can be used according to the requirements. If we compare AI based with engineering method, then it requires less detailed physical information and thus results in low cost and less time. But if we consider AI-based method in real practice, it has no relation between physical building parameters and model inputs, requires training data for model establishment, and the most important is, it needs to be re-trained if changes are made to the building envelope, system or operation. Hence, we can’t say which model is best, it mainly depends on the requirements and the availability of the dataset.
