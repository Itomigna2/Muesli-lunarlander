# Develope inside the docker
FROM ubuntu:22.04

ARG git_config_name=""
ARG git_config_email=""

RUN apt-get update && apt-get install -y git \
    python3-pip \
    vim \
    bash

RUN pip install jupyterlab
RUN pip install --upgrade jupyterlab jupyterlab-git
RUN pip install jupyter-collaboration
EXPOSE 8888

RUN git config --global user.name ${git_config_name}
RUN git config --global user.email ${git_config_email}


# Lib versions will be fixed


# For avoiding caching when build
ARG CACHEBUST=1
RUN git clone https://github.com/Itomigna2/Muesli-lunarlander.git


CMD ["jupyter", "lab", "--ip='*'", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/Muesli-lunarlander"]


# Docker commands:
#  In your Dockerfile directory
#    docker build --build-arg git_config_name="your_git_name" --build-arg git_config_email="your_git_email" --build-arg CACHEBUST=$(date +%s) -t muesli_image .
# docker run --gpus '"device=0,1"' -p 8888:8888 --name mu --rm -it muesli_image
#    (adjust gpus for your configuration)
# access http://your.server.ip:8888 through your browser with token
# Ctrl + P,Q to make docker bg
# use it


