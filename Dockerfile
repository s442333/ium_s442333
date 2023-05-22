FROM ubuntu:latest

# Run apt-get, stop if anything fails.
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python3-pip wget git && \
    apt-get autoremove -y && \
    apt-get clean

RUN python3 -m pip install torch
RUN python3 -m pip install kaggle pandas onnx onnx2torch matplotlib

WORKDIR /scripts

COPY *.sh /scripts/
COPY *.py /scripts/

ENV CUTOFF=100000

CMD ['sh', 'prepare-data.sh']