# Flask Libraries
from flask import redirect, Flask, render_template, json, request, session, url_for, jsonify
from flask.ext.mysql import MySQL
from werkzeug import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_restful import Resource, Api
import jinja2
env = jinja2.Environment()
env.globals.update(zip=zip)
import argparse

# Boto aws connection

# adding parse
parser = argparse.ArgumentParser()
parser.add_argument("--awsid",help="put your amazon access keys")
parser.add_argument("--awskey",help="put your amazon secret access key")
parser.add_argument("--mysqlpass",help="put your amazon secret access key")
parser.add_argument("--s3loc",help="put the region you want to select for amazon s3")

args=parser.parse_args()


import boto
import sys, os
from boto.s3.key import Key

if False:
	LOCAL_PATH = 'static/'
	AWS_ACCESS_KEY_ID = args.awsid
	AWS_SECRET_ACCESS_KEY = args.awskey

	bucket_name = 'alphabetagamma-assignment3'
	# connect to the bucket
	conn = boto.connect_s3(AWS_ACCESS_KEY_ID,
	                AWS_SECRET_ACCESS_KEY)
	bucket = conn.get_bucket(bucket_name)
	# go through the list of files
	bucket_list = bucket.list()
	for l in bucket_list:
	  keyString = str(l.key)
	  l.get_contents_to_filename(LOCAL_PATH+keyString)


app = Flask(__name__)

mysql = MySQL()
# MySQL configurations
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = args.mysqlpass
app.config['MYSQL_DATABASE_DB'] = 'adsingps'
app.config['MYSQL_DATABASE_HOST'] = '198.199.74.32'
mysql.init_app(app)

conn = mysql.connect()
cursor = conn.cursor()

#---------------------------------------------------------------------
# Machine Leanring Stuff

# Data Collection and Transformations
import numpy as np
import pandas as pd
import datetime as dt
import time
import pickle
from sklearn.preprocessing import Imputer, StandardScaler


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
# Plotting 


def predict_single(data,model):
	pred = {}
	if(model == "RandomForest"):
		filename = "static/models/RandomForestRegressor.pckl"
		loaded_model = pickle.load(open(filename, 'rb'))
		pred['latlot'] = loaded_model.predict(data.reshape(1, -1))
		filename = "static/models/RandomForestClassifier.pckl"
		loaded_model = pickle.load(open(filename, 'rb'))
		pred['location'] = loaded_model.predict(data.reshape(1, -1))
	if(model == "ExtraTrees"):
		filename = "static/models/ExtraTreesRegressor.pckl"
		loaded_model = pickle.load(open(filename, 'rb'))
		pred['latlot'] = loaded_model.predict(data.reshape(1, -1))
		filename = "static/models/ExtraTreesClassifier.pckl"
		loaded_model = pickle.load(open(filename, 'rb'))
		pred['location'] = loaded_model.predict(data.reshape(1, -1))
	# if(model == "KNeighbors"):
	# 	filename = "static/models/KNeighborsRegressor.pckl"
	# 	loaded_model = pickle.load(open(filename, 'rb'))
	# 	pred['latlot'] = loaded_model.predict(data.reshape(1, -1))
	# 	filename = "static/models/KNeighborsClassifier.pckl"
	# 	loaded_model = pickle.load(open(filename, 'rb'))
	# 	pred['location'] = loaded_model.predict(data.reshape(1, -1))
	return pred

def predict_single_all(data):
	pred = {}
	filename = "static/models/RandomForestRegressor.pckl"
	loaded_model = pickle.load(open(filename, 'rb'))
	pred['latlot_RandomForest'] = loaded_model.predict(data.reshape(1, -1))
	filename1 = "static/models/RandomForestClassifier.pckl"
	loaded_model1 = pickle.load(open(filename1, 'rb'))
	pred['location_RandomForest'] = loaded_model1.predict(data.reshape(1, -1))
	filename2 = "static/models/ExtraTreesRegressor.pckl"
	loaded_model2 = pickle.load(open(filename2, 'rb'))
	pred['latlot_ExtraTrees'] = loaded_model2.predict(data.reshape(1, -1))
	filename3 = "static/models/ExtraTreesClassifier.pckl"
	loaded_model3 = pickle.load(open(filename3, 'rb'))
	pred['location_ExtraTrees'] = loaded_model3.predict(data.reshape(1, -1))
	filename4 = "static/models/KNeighborsRegressor.pckl"
	loaded_model4 = pickle.load(open(filename4, 'rb'))
	# pred['latlot_KNeighbors'] = loaded_model4.predict(data.reshape(1, -1))
	# filename5 = "static/models/KNeighborsClassifier.pckl"
	# loaded_model5 = pickle.load(open(filename5, 'rb'))
	# pred['location_KNeighbors'] = loaded_model5.predict(data.reshape(1, -1))
	return pred

def predictDf(data,model):
	df = data.drop(['FLOOR', 'BUILDINGID','SPACEID','LONGITUDE','LATITUDE','RELATIVEPOSITION','USERID','PHONEID','TIMESTAMP'], axis=1)
	filename = "static/models/RandomForestRegressor.pckl"
	pred = {}
	if(model == "RandomForest"):
		filename = "static/models/RandomForestRegressor.pckl"
		loaded_model = pickle.load(open(filename, 'rb'))
		pred['latlot'] = loaded_model.predict(df)
		filename = "static/models/RandomForestClassifier.pckl"
		loaded_model = pickle.load(open(filename, 'rb'))
		pred['location'] = loaded_model.predict(df)
	if(model == "ExtraTrees"):
		filename = "static/models/ExtraTreesRegressor.pckl"
		loaded_model = pickle.load(open(filename, 'rb'))
		pred['latlot'] = loaded_model.predict(df)
		filename = "static/models/ExtraTreesClassifier.pckl"
		loaded_model = pickle.load(open(filename, 'rb'))
		pred['location'] = loaded_model.predict(df)
	# if(model == "KNeighbors"):
	# 	filename = "static/models/KNeighborsRegressor.pckl"
	# 	loaded_model = pickle.load(open(filename, 'rb'))
	# 	pred['latlot'] = loaded_model.predict(df)
	# 	filename = "static/models/KNeighborsClassifier.pckl"
	# 	loaded_model = pickle.load(open(filename, 'rb'))
	# 	pred['location'] = loaded_model.predict(df)
	return pred

def predictDf_all(data):
	df = data.drop(['FLOOR', 'BUILDINGID','SPACEID','LONGITUDE','LATITUDE','RELATIVEPOSITION','USERID','PHONEID','TIMESTAMP'], axis=1)
	filename = "static/models/RandomForestRegressor.pckl"
	pred = {}
	filename = "static/models/RandomForestRegressor.pckl"
	loaded_model = pickle.load(open(filename, 'rb'))
	pred['latlot_RandomForest'] = loaded_model.predict(df)
	filename = "static/models/RandomForestClassifier.pckl"
	loaded_model = pickle.load(open(filename, 'rb'))
	pred['location_RandomForest'] = loaded_model.predict(df)
	filename = "static/models/ExtraTreesRegressor.pckl"
	loaded_model = pickle.load(open(filename, 'rb'))
	pred['latlot_ExtraTrees'] = loaded_model.predict(df)
	filename = "static/models/ExtraTreesClassifier.pckl"
	loaded_model = pickle.load(open(filename, 'rb'))
	pred['location_ExtraTrees'] = loaded_model.predict(df)
	# filename = "static/models/KNeighborsRegressor.pckl"
	# loaded_model = pickle.load(open(filename, 'rb'))
	# pred['latlot_KNeighbors'] = loaded_model.predict(df)
	# filename = "static/models/KNeighborsClassifier.pckl"
	# loaded_model = pickle.load(open(filename, 'rb'))
	# pred['location_KNeighbors'] = loaded_model.predict(df)
	return pred

#---------------------------------------------------------------------
# All Routes

@app.route("/")
def main():
	print("lol")
	data = {}
	data['title'] = "Home Page"
	return render_template('index.html',data=data)

@app.route("/register")
def register(message=""):
	data = {}
	data['title'] = "Register"
	data['message'] = message
	return render_template('register.html', data=data)

@app.route('/signup',methods=['POST'])
def signup():
	print("Abhi")
	# read the posted values from the UI
	if request.method == 'POST':
		print("Abhi")
		_name = request.form['inputName']
		_email = request.form['inputEmail']
		_password = request.form['inputPassword']
		_hashed_password = generate_password_hash(_password)
		print(_hashed_password)
		print("Abhi")
		# validate the received values
		cursor.callproc('sp_createUser',(_name,_email,_hashed_password))
		data = cursor.fetchall()
		if len(data) is 0:
			conn.commit()
			data2 = {}
			data2['message'] = "Succuessfully Registered"
			return render_template('login_new2.html', data=data2)
		else:
			return redirect("/register")

@app.route('/login', methods=['GET', 'POST'])
def login(message=""):
	data = {}
	data['message'] = message
	error = None
	if 'user_name' in session:
		return redirect("/dashboard")
	if request.method == 'POST':
		data = {}
		data['title'] = "Login Page"
		username_form  = request.form['inputEmail']
		password_form  = request.form['inputPassword']

		cursor.execute("SELECT * FROM tbl_user WHERE user_username = %s;", [username_form]) # CHECKS IF USERNAME EXSIST
		
		k = cursor.fetchone()
		if (k):
			cursor.execute("SELECT * FROM tbl_user WHERE user_username = %s;", [username_form])# FETCH THE HASHED PASSWORD
			y =  cursor.fetchall()
			print(y)
			for row in y:
				if check_password_hash(row[3], password_form):
					session['userEmail'] = request.form['inputEmail']
					session['userName'] = row[1]
					session['userId'] = row[0]
					data['message'] = "signedin"
					return redirect("/dashboard")
				else:
					data['message'] = "Invalid Credential"
					return render_template('login_new2.html', data=data)
		else:
			error = "Invalid Credential"
	return render_template('login_new2.html', data=data)

@app.route('/logout')
def logout():
	session.pop('userEmail', None)
	session.pop('userName', None)
	session.pop('userId', None)
	print(session)
	return redirect("/login")

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
	data = {}
	data['auth'] = authLevel()
	return render_template('dash_home.html', data=data)

@app.route('/singledata', methods=['GET','POST'])
def singledata():
	data = {}
	data['auth'] = authLevel()
	data['pred'] = ""
	data['isResults'] = False
	if request.method == 'POST':
		formdata  = request.form['inputdata']
		model  = request.form['model']
		
		wapArray = formdata.split(",")
		wapArray = list(map(int, wapArray))
		wapArray = np.asarray(wapArray)
		data['pred'] = predict_single(wapArray,model)
		data['isResults'] = True
		return render_template('upload_single.html', data=data, model=model)
	return render_template('upload_single.html', data=data)

def allowed_file(filename):
	return '.' in filename and \
		filename.rsplit('.', 1)[1].lower() in ['csv','excel']

@app.route('/singlefile', methods=['GET','POST'])
def singlefile():
	data = {}
	data['auth'] = authLevel()
	data['pred'] = ""
	data['filename'] =""
	data['isResults'] = False
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			print('No file part')
			return redirect(request.url)
		file = request.files['file']
		model  = request.form['model']
		# if user does not select file, browser also
		# submit a empty part without filename
		if file.filename == '':
			print('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			#file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			#return redirect(url_for('uploaded_file',filename=filename))
			data['filename'] = filename
			trainingData= pd.read_csv(file)
			data['results'] = predictDf(trainingData,model)
			data_all = []
			for k,i in zip(data['results']['latlot'],data['results']['location']):
				r = [k,i]
				data_all.append(r)
			print(data_all)
			data['resultsArray'] = np.asarray(data['results'])
			data['isResults'] = True
			print(data['results'])
			return render_template('upload_file.html', data=data)
	return render_template('upload_file.html', data=data)

#-------------------------------------------------------------------
# API definition

@app.route("/getLocation", methods=["POST"])
def getLocation():
	if request.json['wapData']:
		parameters = request.json['wapData']
		wapArray = parameters.split(",")
		if len(wapArray) == 520:
			wapArray = list(map(int, wapArray))
			wapArray = np.asarray(wapArray)
			data = {}
			data['pred'] = predict_single_all(wapArray)
			return str(data['pred'])
		else:
			return "Incorrect number of parameters \nPlease send in 520 wap position in order of wap001 to wap520"
	else:
		return "Wrong Parameter passed"

#-------------------------------------------------------------------
# Helper function for authentication

def isAdmin():
	cursor.execute("SELECT * FROM userauth WHERE userid = %s AND authid=1;", session['userId'])# FETCH THE HASHED PASSWORD
	query = cursor.fetchone()
	if (query):
		return True
	else:
		return False

def isCompany():
	cursor.execute("SELECT * FROM userauth WHERE userid = %s AND authid=2;", session['userId'])# FETCH THE HASHED PASSWORD
	query = cursor.fetchone()
	if (query):
		return True
	else:
		return False

def isSingle():
	cursor.execute("SELECT * FROM userauth WHERE userid = %s AND authid=3;", session['userId'])# FETCH THE HASHED PASSWORD
	query = cursor.fetchone()
	if (query):
		return True
	else:
		return False

def authLevel():
	auth = {}
	auth['userName'] = session['userName']
	if(isAdmin()):
		auth['name'] = "Administrator"
		auth['type'] = "admin"
		return auth
	elif(isCompany()):
		auth['name'] = "Company"
		auth['type'] = "comp"
		return auth
	elif(isSingle()):
		auth['name'] = "Single User"
		auth['type'] = "single"
		return auth


app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'


if __name__ == "__main__":
	app.run(host='0.0.0.0')

