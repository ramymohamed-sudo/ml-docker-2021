FROM python:latest
# smizy/scikit-learn


WORKDIR /var/lib/python

COPY . . 

RUN pip install --no-cache-dir -r requirements.txt
# RUN /usr/local/bin/python -m pip install --upgrade pip
# You should consider upgrading via the '/usr/bin/python3.8 -m pip install --upgrade pip' command.

# COPY train.csv ./train.csv
# COPY test.csv ./test.csv
# COPY train.py ./train.py
# COPY inference.py ./inference.py

# RUN python3 train.py # this will run in the run stage in Jenkins

CMD [ "python3", "./lr_rf_svr.py" ]
# docker build --tag classical-reg .
# docker run -d -v /home/k8s/minikube/dockers-images/ml_dockers/classical-reg/logs-h5-models:/var/lib/python/logs-h5-models classical-reg