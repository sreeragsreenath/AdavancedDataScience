Pipeline
Luigi 
The purpose of Luigi is to address all the plumbing typically associated with long-running batch processes.
Conceptually, Luigi is similar to GNU Make where you have certain tasks and these tasks in turn may have dependencies on other tasks
The Luigi server comes with a web interface too, so you can search and filter among all your tasks.
Steps to Perform for Luigi
Requirement :  Python 2.7.*
Run pip install luigi to install Luigi
Run the build.sh, it will create a virtual environment in the root directory of the project with all the dependencies.
It will also setup the luigi Central Scheduler and run it as a daemon process.
The Luigi Task Visualiser can be accessed by http://localhost:8082 which will give visualisation of all the running tasks.
Python Object Serialization
Pickle
It is used for serializing and de-serializing a Python object structure. 
Any object in python can be pickled so that it can be saved on disk. 
What pickle does is that it “serialize” the object first before writing it to file. Pickling is a way to convert a python object (list, dict, etc.) into a character stream. 
The idea is that this character stream contains all the information necessary to reconstruct the object in another python script.

Dockerized
Docker
Docker is an open source tool that automates the deployment of the application inside software container. 
When you develop an application, you need to provide your code alongside with all possible dependencies like libraries, web server, databases, etc. You may end up in a situation when the application is working on your computer but won’t even start on stage server, dev or a QA’s machine.
This challenge can be addressed by isolating the app to make it independent of the system.
Steps to Perform for Docker
We have used an Ubuntu 16.4 image and updated its libraries. 
Then we installed Python3 and Python pip to execute the python script for web scraping.
Required libraries has been stated in the dockerfiles itself which include : ● Boto

Images of our process
Model Deployment
Web Application using Flask
Requirement steps : 
pip install flask
pip install flask_mysql
pip install flask_restful
Python app.py
Start the Apache server and connect it with mySql using Xamp
Steps to perform for REST Api calls using JSON
Requirement : Install Advanced REST Client


Amazon S3
Amazon Simple Storage Service (Amazon S3) is storage for the Internet.
You can use Amazon S3 to store and retrieve any amount of data at any time, from anywhere on the web.
