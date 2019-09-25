ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:19.05-py3
FROM ${FROM_IMAGE_NAME}

RUN apt-get update && apt-get install -y pbzip2 pv bzip2 openssh-server

RUN pip install toposort networkx pytest nltk tqdm html2text progressbar pyparsing

WORKDIR /workspace
# RUN git clone https://github.com/openai/gradient-checkpointing.git
# RUN git clone https://github.com/attardi/wikiextractor.git
# RUN git clone https://github.com/soskek/bookcorpus.git

WORKDIR /workspace/bert
# COPY . .

ENV PYTHONPATH=/workspace/bert

# Fix ssh for passwordless connection between containers.
RUN ssh-keygen -t rsa -N "" -f /root/.ssh/id_rsa
RUN cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
RUN ssh-keyscan -H localhost >> /root/.ssh/known_hosts
RUN echo "Host *\n\tStrictHostKeyChecking no" >> /root/.ssh/config && chmod 600 /root/.ssh/config

RUN echo 'ALL ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# apt update && apt install -y --no-install-recommends wget perl python curl lsb-release
# tar -xzvf MLNX_OFED_LINUX-4.1-1.0.2.0-ubuntu16.04-x86_64.tgz &&    MLNX_OFED_LINUX-4.1-1.0.2.0-ubuntu16.04-x86_64/mlnxofedinstall --user-space-only --without-fw-update --all -q &&    cd .. &&    rm -rf MLNX_OFED_LINUX-4.1-1.0.2.0-ubuntu16.04-x86_64 &&    rm -rf *.tgz