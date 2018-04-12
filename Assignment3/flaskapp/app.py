from flask import redirect, Flask, render_template, json, request, session, url_for
from flask.ext.mysql import MySQL
from werkzeug import generate_password_hash, check_password_hash

app = Flask(__name__)

mysql = MySQL()
# MySQL configurations
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'adsingps'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql.init_app(app)

conn = mysql.connect()
cursor = conn.cursor()

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
	app.run()