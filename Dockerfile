FROM caffe2/caffe2:snapshot-py2-cuda9.0-cudnn7-ubuntu16.04

RUN mv /usr/local/caffe2 /usr/local/caffe2_build
ENV Caffe2_DIR /usr/local/caffe2_build
ENV PYTHONPATH /usr/local/caffe2_build:${PYTHONPATH}
ENV LD_LIBRARY_PATH /usr/local/caffe2_build/lib:${LD_LIBRARY_PATH}
RUN mkdir /PTF
# Clone the Detectron repository
RUN git clone https://github.com/facebookresearch/densepose /densepose

# Install Python dependencies
RUN pip install -r /densepose/requirements.txt

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

#Densepose pre-install
RUN apt -yq update
RUN apt install -yq wget nano ffmpeg
WORKDIR ./DensePoseData
RUN bash get_densepose_uv.sh
RUN pip2 install "pyyaml==3.12"
# Need to change the config
# cd configs
# cp DensePose_ResNet101_FPN_s1x-e2e.yaml my_config.yaml my_config.yaml
# nano my_config.yaml
# Change line 77 Dectections_per_im to 1.

# MY STUFF
WORKDIR /PTF
ENV DEBIAN_FRONTEND noninteractive
RUN apt -yq update
RUN apt install -yq software-properties-common
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt -yq update
RUN apt -yq install python3.6
RUN apt -yq install ffmpeg
RUN curl https://bootstrap.pypa.io/get-pip.py | sudo -H python3.6
RUN git clone https://github.com/akanazawa/human_dynamics ./hmmr
RUN git clone https://github.com/poxyu/punch-to-face-tech ./ptf
# Install Python dependencies
ENV CUDA_HOME /usr/local/cuda
RUN export CUDA_HOME=/usr/local/cuda
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 10
RUN pip3.6 install -r ./hmmr/requirements.txt
WORKDIR ./hmmr/src/external
RUN apt install -yq python3-tk python3.6-dev
RUN bash install_external.sh
WORKDIR ./neural_renderer
ENV CUDA_HOME /usr/local/cuda
RUN export CUDA_HOME=/usr/local/cuda
WORKDIR /PTF/hmmr
RUN wget http://angjookanazawa.com/cachedir/hmmr/hmmr_models.tar.gz && tar -xf hmmr_models.tar.gz
RUN wget http://angjookanazawa.com/cachedir/hmmr/hmmr_demo_data.tar.gz && tar -xf hmmr_demo_data.tar.gz
WORKDIR /PTF/hmmr/src/external/AlphaPose/models/yolo
RUN wget https://pjreddie.com/media/files/yolov3-spp.weights
WORKDIR /PTF/hmmr/src/external/AlphaPose/models/yolo
RUN wget https://pjreddie.com/media/files/yolov3-spp.weights
WORKDIR /PTF/hmmr/src/external/AlphaPose/models/sppe
RUN wget https://al-deeplearn.s3.amazonaws.com/duc_se.pth
#RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10
# Need to Run hmmr/src/external/install_dependencies.sh
# Running poxyu notebook
WORKDIR /PTF
RUN pip3.6 install ffmpy jupyter "opencv-python==3.4.2.16" --force-reinstall


