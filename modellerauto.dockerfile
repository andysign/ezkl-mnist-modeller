# docker build -t zk-demo-modellerauto -f modellerauto.dockerfile .

FROM python:3.10-bookworm@sha256:d47cb1b09be409a0cb5efeb01dd3ff5d44a5e28d371fa1a736c8928ab72fffd2

# Create the app directory in the container and the models directory
RUN mkdir /app

# Set the working directory
WORKDIR /app

# Enable venv
ENV PATH="/app/venv/bin:$PATH"

# Install packages, start python virtual environment
RUN python -m venv venv
RUN /bin/bash -c "source /app/venv/bin/activate"
RUN pip install --upgrade pip
RUN pip install torch==2.3.1 torchvision==0.18.1 tf2onnx==1.16.1 onnx==1.16.1

# Copy over the main Python mnist file
COPY ./app/mnist.py /app/mnist.py

# Expose the needed port
EXPOSE 8081

# Check the health of the container
HEALTHCHECK CMD curl --fail http://localhost:8081 || exit 1

# The main entrypoint for docker
ENTRYPOINT ["sh", "-c", "python mnist.py && python -m http.server 8081"]

# docker run --rm -it --name zk-modellerauto -h zk-modellerauto -p 8081:8081 --entrypoint /bin/bash zk-demo-modellerauto

# docker run --rm -it --name zk-modellerauto -h zk-modellerauto -p 8081:8081 -v "$PWD/mnist-train-n-test.tar.gz":/app/mnist-train-n-test.tar.gz -v "$PWD/models-pkl.tar.gz":/app/models-pkl.tar.gz zk-demo-modellerauto

# docker run --rm -it --name zk-modellerauto -h zk-modellerauto -p 8081:8081 zk-demo-modellerauto

# is_healthy() { if [[ "$(docker inspect --format='{{.State.Health.Status}}' zk-modellerauto 2> /tmp/err)" == "healthy" ]]; then echo "healthy"; else echo ""; fi; };

# echo -en "\n\nWaiting..."; RES=""; while [[ -z "$RES" ]]; do sleep .1; RES=$(is_healthy); echo -n .; done; echo;
