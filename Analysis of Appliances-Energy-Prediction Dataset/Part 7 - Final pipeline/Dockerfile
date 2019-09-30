FROM ubuntu:16.04


RUN mkdir -p /usr/src/team1assignment2
WORKDIR /usr/src/team1assignment2/

RUN apt-get update -y
RUN apt-get install python3 python3-pip -y


COPY final_pipeline.py /usr/src/team1assignment2/
COPY energydata_complete.csv /usr/src/team1assignment2/


RUN pip3 install pandas
RUN pip3 install boto3
RUN pip3 install sklearn
RUN pip3 install scipy


CMD ["sh","-c","python3 /usr/src/team1assignment2/final_pipeline.py "] 