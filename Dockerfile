# Use Caffe2 image as parent image
FROM caffe2/caffe2:snapshot-py2-cuda9.0-cudnn7-ubuntu16.04

RUN mv /usr/local/caffe2 /usr/local/caffe2_build
ENV Caffe2_DIR /usr/local/caffe2_build

ENV PYTHONPATH /usr/local/caffe2_build:${PYTHONPATH}
ENV LD_LIBRARY_PATH /usr/local/caffe2_build/lib:${LD_LIBRARY_PATH}

# Clone the Detectron repository and HMMR
ENV DEBIAN_FRONTEND noninteractive
RUN apt -yq  update
RUN apt -yq  install python3.6.1-pip
RUN git clone https://github.com/facebookresearch/densepose /densepose
RUN git clone https://github.com/akanazawa/human_dynamics /hmmr
RUN git clone https://github.com/poxyu/punch-to-face-tech /ptf
# Install Python dependencies
RUN pip install -r /densepose/requirements.txt
RUN pip3 install -r /hmmr/requirements.txt
RUN pip3 install jupyter "opencv-python==3.4.2.16" --force-reinstall
# Install the COCO API
RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi
WORKDIR /cocoapi/PythonAPI
RUN make install

# Go to Densepose root
WORKDIR /densepose

# Set up Python modules
RUN make

# [Optional] Build custom ops
RUN make ops
