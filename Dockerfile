ARG PARENT_IMAGE
FROM $PARENT_IMAGE
ARG PYTHON_VERSION=3.9.15

RUN apt-get update && \
    echo Y | apt-get install vim

ENV CODE_DIR=/home/user

WORKDIR ${CODE_DIR}

# make container user to be the same as host user (non-root user)
# default UID and GID are 1000 for general user
ARG UID=1000    
ARG GID=1000
RUN groupadd -g $GID mygroup && \
    useradd -u $UID -g mygroup -m myuser
# change default user from root to myuser
USER myuser

RUN pip3 install --upgrade pip && \
    pip install stable-baselines3[extra]==2.3.2 && \
    pip install tensorboard==2.18.0 && \
    pip install pyyaml==6.0.2 && \
    pip install scipy==1.13.0 && \
    pip install wandb==0.18.7 && \
    pip install numpy==1.26.3 && \
    pip cache purge

CMD ["/bin/bash"]