# Develope inside the docker
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y git \
    python3-pip \
    vim \
    bash

RUN pip install jupyterlab
#RUN jupyter lab --generate-config \
#    && echo "c.NotebookApp.terminado_settings = {'shell_command': ['/bin/bash']}" >> ~/.jupyter/jupyter_notebook_config.py

EXPOSE 8888

# Lib versions will be fixed


# For avoiding caching when build
ARG CACHEBUST=1
RUN git clone https://github.com/Itomigna2/Muesli-lunarlander.git


CMD ["jupyter", "lab", "--ip='*'", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/Muesli-lunarlander"]




# Docker commands:
#  In your Dockerfile directory
#    docker build --build-arg CACHEBUST=$(date +%s) -t muesli_image .
# docker run --gpus '"device=0,1"' -p 8888:8888 --name mu --rm -it muesli_image
# access https://your.server.ip:8888 through your browser with token
# Ctrl + P,Q to make docker bg
# use it


