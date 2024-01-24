# Develope inside the docker
FROM ubuntu:22.04

ARG git_config_name=""
ARG git_config_email=""

RUN apt-get update && apt-get install -y git \
    python3-pip \
    vim \
    ffmpeg \
    libsm6 \
    libxext6    
    
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install python3.10
RUN ln -s /usr/bin/python3.10 /usr/bin/python
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install jupyterlab
RUN pip install --upgrade jupyterlab jupyterlab-git
RUN pip install jupyter-collaboration
EXPOSE 8888

RUN pip install nni
EXPOSE 8080

RUN git config --global user.name ${git_config_name}
RUN git config --global user.email ${git_config_email}


COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install git+https://github.com/cmpark0126/pytorch-polynomial-lr-decay.git
# Lib versions will be fixed


# For avoiding caching when build
ARG CACHEBUST=1
RUN git clone https://github.com/Itomigna2/Muesli-lunarlander.git


CMD ["jupyter", "lab", "--ip='*'", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/Muesli-lunarlander"]


# Docker commands:
#  In your Dockerfile directory
#    docker build --build-arg git_config_name="your_git_name" --build-arg git_config_email="your_git_email" --build-arg CACHEBUST=$(date +%s) -t muesli_image .
# docker run --gpus '"device=0,1"' -p 8888:8888 -p 8080:8080 --name mu --rm -it muesli_image
#    (adjust gpus for your configuration)
# access http://your.server.ip:8888 through your browser with token
# Ctrl + P,Q to make docker bg
# use it

