FROM tensorflow/tensorflow:latest-gpu

WORKDIR /code

COPY requirements.txt .

ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y python3-opencv
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3 -m pip install git+https://github.com/BindsNET/bindsnet.git
Run python3 -m pip install ANNarchy



