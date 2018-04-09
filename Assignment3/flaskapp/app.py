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



@app.route("/")
def main():
	print("lol")
	data = {}
	data['ok'] = "cool"
	data['ok2'] = 'cool2'
	return render_template('index.html',data=data)

@app.route("/showSignUp")
def showSignUp():
	return render_template('signup.html')

@app.route('/signUp',methods=['POST'])
def signUp():

    # read the posted values from the UI
    _name = request.form['inputName']
    _email = request.form['inputEmail']
    _password = request.form['inputPassword']
    _hashed_password = generate_password_hash(_password)
    print(_hashed_password)
    
    # validate the received values
    cursor.callproc('sp_createUser',(_name,_email,_hashed_password))
    data = cursor.fetchall()
    if len(data) is 0:
    	conn.commit()
    	return json.dumps({'message':'User created successfully !'})
    else:
    	return json.dumps({'error':str(data[0])})

@app.route('/login', methods=['GET', 'POST'])
def login():
	error = None
	if 'user_name' in session:
		return redirect("/dashboard")
	if request.method == 'POST':
		username_form  = request.form['inputEmail']
		password_form  = request.form['inputPassword']

		cursor.execute("SELECT * FROM tbl_user WHERE user_username = %s;", [username_form]) # CHECKS IF USERNAME EXSIST
		
		k = cursor.fetchone()
		print(k)
		if (k):
			cursor.execute("SELECT user_password FROM tbl_user WHERE user_username = %s;", [username_form])# FETCH THE HASHED PASSWORD
			y =  cursor.fetchall()
			print(y)
			for row in y:
				print(row[0], " ",password_form)
				if check_password_hash(row[0], password_form):
					session['user_name'] = request.form['inputEmail']
					error = "signedin"
					print(session)
					return redirect("/")
				else:
					error = "Invalid Credential"
		else:
			error = "Invalid Credential"
	return render_template('login_new.html', error=error)

@app.route('/logout')
def logout():
	session.pop('user_name', None)
	print(session)
	return redirect("/login")

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
	return render_template('dashboard.html', error=error)

app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'
if __name__ == "__main__":
	app.run()