FROM ubuntu:16.04



RUN apt-get update -y
RUN apt-get install python3 python3-pip -y


COPY flaskapp/. /flask
RUN pip3 install jupyter
RUN pip3 install pandas
RUN pip3 install boto3
RUN pip3 install numpy
RUN pip3 install boto
RUN pip3 install flask
RUN pip3 install flask_mysql
RUN pip3 install flask_restful
RUN pip3 install sklearn
RUN pip3 install scipy
RUN apt-get install nano -y

ENV awsid 1
ENV awskey 1
ENV mysqlpassn 1
ENV s3loc 1

CMD ["sh","-c","python3 /flask/app.py --mysqlpass ${mysqlpassn}"] 


