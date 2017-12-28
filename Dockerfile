FROM ultron1173/dlib:latest

ADD ./src /opt/server

WORKDIR /opt/server

RUN pip3 install -r requirements.txt

CMD python3 keep_alive.py & gunicorn --bind 0.0.0.0:$PORT start:app