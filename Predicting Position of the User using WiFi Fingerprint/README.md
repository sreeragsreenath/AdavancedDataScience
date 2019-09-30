# Assignment 3

## Introduction

Automatic user localization consists of estimating the position of the user (latitude, longitude and altitude) by using an electronic device, usually a mobile phone. Outdoor localization problem can be solved very accurately thanks to the inclusion of GPS sensors into the mobile devices. However, indoor localization is still an open problem mainly due to the loss of GPS signal in indoor environments. With the widespread use of Wi-Fi communication in indoor environments, Wi-Fi or wireless local area network (WLAN) based positioning gained popularity to solve indoor localization.
WLAN-based positioning systems utilize the Wi-Fi received signal strength indicator (RSSI) value. In this project, we focus on fingerprint-based localization. Fingerprinting technique consists of two phases: calibration and positioning. In the calibration phase, an extensive radio map is built consisting of RSSI values from multiple Wi-Fi Access Points (APs) at different knownlocations. This calibration data is used to train the localization algorithm. In the positioning phase, when a user reports the RSSI measurements for the multiple APs, the fit algorithm predicts the user position.
A key challenge in wireless localization is that RSSI value at a given location can have large fluctuations due to Wi-Fi interference, user mobility, environmental mobility etc. In this project, we design, implement and evaluate machine learning algorithms for WLAN fingerprint-based localization.

 
Dataset Description 
Source: https://www.kaggle.com/giantuji/UjiIndoorLoc
WAP001-WAP520: Intensity value for Wireless Access Point (AP). AP will be the acronym used for rest of this notebook. Negative integer values from -104 to 0 and +100. Censored data: Positive value 100 used if WAP was not detected.
Longitude: Longitude. Negative real values from -7695.9387549299299000 to -7299.786516730871000
Latitude: Latitude. Positive real values from 4864745.7450159714 to 4865017.3646842018.
Floor: Altitude in floors inside the building. Integer values from 0 to 4.
BuildingID: ID to identify the building. Measures were taken in three different buildings. Categorical integer values from 0 to 2.
SpaceID: Internal ID number to identify the Space (office, corridor, classroom) where the capture was taken. Categorical integer values.
RelativePosition: Relative position with respect to the Space (1 - Inside, 2 - Outside in Front of the door). Categorical integer values.
UserID: User identifier (see below). Categorical integer values.
PhoneID: Android device identifier (see below). Categorical integer values.
Timestamp: UNIX Time when the capture was taken. Integer value

Project Tools:
Language : Python 
Pipeline : Luigi
Framework : Flask
Databse : SQL
Tools used: Jupyter Notebooks and Docker , Xamp. 

Exploratory Data Analysis 
Conduct EDA using Python libraries - 
Seaborn, Numpy, MissingNo, Pickle, Plotly, Pandas, Scipy, Sklearn, Matplotlib			
Provide a PowerPoint Report with graphs and key insights. 

Prediction Algorithms -
Random Forest, ExtraTrees models in using sklearn in Python. Compute RMS, MAPE, R2 and MAE for Training and Testing Datasets. Recommend a model. 

Model Validation and Selection - Used Crossvalidation


Best Model:
Extra Trees Classifier - 86% r -square
Extra Trees Regressor - 98% r-square

### INFO
1. Language used : Python
2. Process Followed : Data Ingestion, Data Wrangling, Exploratory Data Analysis
3. Tools used :  Jupyter Notebook


