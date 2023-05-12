 # got this from hub.docker
FROM nvidia/cudagl:11.3.1-devel-ubuntu20.04

ENV VIRTUALGL_VERSION 2.5.2
ARG NUM_BUILD_CORES

# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# Install all apt dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
RUN apt-get update && apt-get install -y software-properties-common
# Add ppa for python3.8 install
RUN apt-add-repository -y ppa:deadsnakes/ppa
RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ="America/New_York" \
	apt-get install -y --no-install-recommends \
	curl ca-certificates cmake git python3.8 python3.8-dev \
	xvfb g++ \
	libboost-all-dev libcgal-dev ffmpeg python3.8-tk \
	libeigen3-dev
	

# Install openGL (the mesa-.. package is not compitable with ubuntu20)
RUN apt-get update && apt-get install -y libglfw3-dev libgl1-mesa-dev \
				libglu1-mesa-dev freeglut3-dev libgl1-mesa-dri

# Install VirtualGL
RUN curl -sSL https://downloads.sourceforge.net/project/virtualgl/"${VIRTUALGL_VERSION}"/virtualgl_"${VIRTUALGL_VERSION}"_amd64.deb -o virtualgl_"${VIRTUALGL_VERSION}"_amd64.deb && \
	dpkg -i virtualgl_*_amd64.deb && \
	/opt/VirtualGL/bin/vglserver_config -config +s +f -t && \
	rm virtualgl_*_amd64.deb


# Install python dependencies
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py && rm get-pip.py

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN echo "backend: TkAgg" > matplotlibrc


# Copy the remaining code
COPY scripts scripts
COPY src/unitybridge unitybridge
COPY src/tests tests


# Set up the starting point for running the code
COPY src/entrypoint.sh /entrypoint.sh
RUN chmod 755 /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]


# copy the cpp code
COPY src/pycpp_examples pycpp_examples
RUN pip3 install ./pycpp_examples
RUN g++ pycpp_examples/display_window.cpp -lGL -lglut -o test




