FROM python:3.6.4-slim-stretch

RUN apt-get -y update
RUN apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    software-properties-common \
    zip \
    sudo \
    python3-pip \
    libopenblas-dev \
    && apt-get clean && rm -rf /tmp/* /var/tmp/* \
    && pip3 install numpy

RUN cd ~ && \
    mkdir -p dlib && \
    git clone -b 'v19.9' --single-branch https://github.com/davisking/dlib.git dlib/ && \
    cd  dlib/ && \
    python3 setup.py install --yes USE_AVX_INSTRUCTIONS

ADD ./src /opt/server

WORKDIR /opt/server

RUN pip3 install -r requirements.txt

CMD python3 keep_alive.py & gunicorn --bind 0.0.0.0:$PORT start:app
