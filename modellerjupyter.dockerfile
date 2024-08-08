# docker build -t zk-demo-modellerjupyter -f modellerjupyter.dockerfile . # OR --progress=plain # DOCKER_CLI_HINTS=false

FROM python:3.10-bookworm@sha256:d47cb1b09be409a0cb5efeb01dd3ff5d44a5e28d371fa1a736c8928ab72fffd2

# Add a jupyter user and change the current user to jupyter
RUN useradd -ms /bin/bash jupyter && passwd -d jupyter
USER jupyter

# Set the working directory
WORKDIR /jupyter

# Enable venv
ENV PATH="/jupyter/venv/bin:$PATH"

# Install packages, start python virtual environment
RUN python -m venv venv
RUN /bin/bash -c "source /jupyter/venv/bin/activate"
RUN pip install --upgrade pip
RUN pip install torch==2.3.1 torchvision==0.18.1 tf2onnx==1.16.1 onnx==1.16.1 matplotlib==3.9.0 jupyter==1.0.0

# Copy over the main Jupyter notebook file
COPY --chown=jupyter:jupyter ./jupyter/mnist.ipynb /jupyter/mnist.ipynb

# Expose the needed port
EXPOSE 8888

# Check the health of the container OR HEALTHCHECK --interval=5s --timeout=5s --start-period=3s
HEALTHCHECK CMD curl --fail http://localhost:8888 || exit 1

# The main entrypoint for docker # ENTRYPOINT jupyter notebook --ip=0.0.0.0 --no-browser --NotebookApp.token='' --NotebookApp.password=''
ENTRYPOINT ["jupyter","notebook","--ip=0.0.0.0","--no-browser","--NotebookApp.token=''","--NotebookApp.password=''"]

# docker run --rm -it --name zk-modellerjupyter -h zk-modellerjupyter -p 8888:8888 --entrypoint /bin/bash zk-demo-modellerjupyter

# docker run --rm -it --name zk-modellerjupyter -h zk-modellerjupyter -p 8888:8888 -v "$PWD/mnist-train-n-test.tar.gz":/jupyter/mnist-train-n-test.tar.gz -v "$PWD/models-pkl.tar.gz":/jupyter/models-pkl.tar.gz zk-demo-modellerjupyter

# docker run --rm -it --name zk-modellerjupyter -h zk-modellerjupyter -p 8888:8888 zk-demo-modellerjupyter

# is_healthy() { if [[ "$(docker inspect --format='{{.State.Health.Status}}' zk-modellerjupyter  2> /tmp/err)" == "healthy" ]]; then echo "healthy"; else echo ""; fi; };

# echo -en "\n\nWaiting..."; RES=""; while [[ -z "$RES" ]]; do sleep .1; RES=$(is_healthy); echo -n .; done; echo;
