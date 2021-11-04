from nvidia/cuda:11.2.0-base-ubuntu20.04
RUN apt update
RUN apt install -y \
    python3 \
    python3-pip
RUN  pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
COPY ./requirements.txt /tmp/requirements.txt 
RUN pip install  -r /tmp/requirements.txt && rm /tmp/requirements.txt
COPY ./ /nas
WORKDIR /nas
ENTRYPOINT sleep 999999999999
