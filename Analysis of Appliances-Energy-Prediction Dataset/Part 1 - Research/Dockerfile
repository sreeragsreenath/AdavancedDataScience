FROM ubuntu:16.04


RUN mkdir -p /usr/src/team1assignment2
WORKDIR /usr/src/team1assignment2/Result

RUN apt-get update -y
RUN apt-get install python3 python3-pip -y


COPY Research_Paper_1/. /usr/src/team1assignment2/
COPY Research_Paper_2/. /usr/src/team1assignment2/
COPY Research_Paper_3/. /usr/src/team1assignment2/
RUN pip3 install jupyter
#RUN pip3 install pandas
#RUN pip3 install boto3
#RUN pip3 install numpy
#RUN pip3 install boto
#RUN pip3 install urllib3
#RUN pip3 install requests
#RUN pip3 install html5lib
#RUN pip3 install bs4
#RUN pip3 install lxml

#ENV ck 1
#ENV ac 1
#ENV akey 1
#ENV skey 1
#ENV sl 1
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension
CMD jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
#CMD ["sh","-c","python /usr/src/team5assignment1/Research_paper_1.py --cik ${ck} --acn ${ac} --akey ${akey} --skey ${skey} --s3loc ${sl}"] 