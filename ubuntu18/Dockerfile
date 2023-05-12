# got this from hub.docker
FROM nvidia/cuda:12.0.0-devel-ubuntu18.04

USER root

ARG EPIC_USER=user
ARG EPIC_PASS=pass
ARG GIT_BRANCH


RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
RUN	apt-get update && apt-get install -y software-properties-common

RUN apt-add-repository -y ppa:deadsnakes/ppa 
RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ="America/New_York" \
	apt-get install -y --no-install-recommends \
	curl ca-certificates cmake git \
	xvfb g++ \
	libboost-all-dev libcgal-dev ffmpeg\
	libeigen3-dev && \
	apt-get update 

# Install python dependencies
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install tqdm

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN apt-get update && apt-get -y install python3-tk


RUN pip3 install PyQt5==5.15.2
# RUN echo "export QT_QPA_PLATFORM=offscreen">> $HOME/profile

# Needed for using matplotlib without a screen
RUN echo "backend: TkAgg" > matplotlibrc

RUN pip3 install torch
RUN pip3 install torchvision
RUN pip3 install torchsummary
RUN pip3 install easydict
RUN pip3 install tensorboard
RUN pip3 install scikit-learn
# for mayavi
RUN pip3 install VTK==8.1.2
RUN pip3 install mayavi
RUN pip3 install shapely

# Set up the starting point for running the code with GPU
COPY /entrypoint.sh /entrypoint.sh
RUN chmod 755 /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Copy the remaining code
# COPY scripts scripts
