# Start with a Linux micro-container to keep the image tiny
FROM python:3.6-slim

# Document who is responsible for this image
MAINTAINER Siddhant Patny "siddhant.patny@nyu.edu"

# Install just the Python runtime (no dev)
# RUN apt-get update \
#  && apt-get install -y python3-pip \
#  && cd /usr/local/bin \
#  && ln -s /usr/bin/python3 python \
#  && pip3 install --upgrade pip


# Expose any ports the app is expecting in the environment
ENV PORT 8001
EXPOSE $PORT

# Set up a working folder and install the pre-reqs
WORKDIR /app
ADD /pytorch-mnist/requirements.txt /app
# ADD /pytorch-mnist/mnist_cnn.pt /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Add the model and code as the last Docker layer because it changes the most
ADD /pytorch-mnist/mnist_cnn.pt /app
ADD /pytorch-mnist/main.py  /app/main.py

# Run the service
CMD [ "python", "main.py" ]

