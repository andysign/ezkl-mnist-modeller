# ZKML MNIST Local Demo

## ZKML MNIST Local Demo: Overview

This project aims to demonstrate how a particular technology that falls under the umbrella term of **Verifiable Compute** works, particularly this project focuses on **Verifiable ML**, also known as **Zero-Knowledge Machine Learning** (abbreviated **ZKML**).

This project aims to demonstrate how a particular technology works, a technology which falls under the umbrella term of **Verifiable Compute** ( or the sub-term, **Verifiable ML** / **Zero-Knowledge Machine Learning** or **ZKML** to be more exact ).

In virtually any ZKML system, **[zero-knowledge proofs](https://en.wikipedia.org/wiki/Zero-knowledge_proof)** will be created alongside the inference process, usually those will be created by a **prover** and then verified by a **verifier** to check the correctness of the computation and for checking if the correct model has been used. For that reason, this repository focuses on training a **good** MNIST model but also training a **bad** MNIST model as well — one could easily prove that with this technology, the switch between the good and bad model can be easily detected ( although, the three services for doing the ZK preparation, the proof creation and the proof verification can be found in a separate repository, namely **[ezkl-mnist-demo](https://github.com/CDECatapult/ezkl-mnist-demo)** ).

It is essentially a project that contains: two services used for training **MNIST** _Convolutional Neural Networks_ that can do basic digit recognition / classification, more exactly, service ONE ( **`modellerjupyter`** ), a service where tha training happens inside a **[Jupyter Notebook](https://jupyter.org/)** and service TWO ( **`modellerauto`** ), a service where the training happens in a vanilla **[Venv](https://docs.python.org/3/library/venv.html)** environment ( **`modellerauto`** ) in an automatic way without the user's intervention.

This is heavily inspired by the **EZKL MNIST Tutorial** available **[here](https://github.com/zkonduit/e2e-mnist/blob/main/mnist_classifier.ipynb)** ( more about that is available below ).

---

## ZKML MNIST Local Demo: About The Underlying ZK Lib

**[EZKL](https://ezkl.xyz/)** (pronounced _Easy KL_) is the underlying library (aka toolkit) that chosen as the base layer for this project, as it seems to be fairly mature and well **[documented](https://docs.ezkl.xyz/)**. **EZKL** helps prove the authenticity of **AI** / **ML** models; it does that by generating a **[zero-knowledge proof](https://xthemadgenius.medium.com/9b7f6ef1c708)** that a model produced certain results, without having to reveal private data or the model itself.

---

## ZKML MNIST Local Demo: Software and Libraries Requirements

<!--

When it comes to the hardware and OS required, it is important to state that this demo requires a **MacBook Pro** ( with **M1** or **M2** cpu ) and it also runs on **MacOs** ( **Sonoma 14.5** ) as no other machines / operating systems have been tested.

In terms of software requirements, **Docker** version **25.0.2** or higher would be highly recommended to have ( **`brew cask install docker`** ).

Also, obviously, **DockerCompose** should also be present, as in, it is highly recommended to have compose version v2.24.3-desktop.1 or higher ( **`sudo -s`** followed by **`curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose`** ).

Another extra small dependency is required, called **JQ**, the widely-used command-line JSON parser, ideally v1.7.1 or higher ( just use **`brew install jq`** to install it ).

**For running the project without virtualisation, the requirements are**:

As we are talking about A.I. models, virtually everything apart from the front-end is written in Python therefor, what's needed is ideally **Python** version **3.10.14** or higher together with **PIP** version **24.1** or higher.

Since, the front-end is written in NextJs, what's needed for the front-end is ideally **NodeJs** version **v20.15.0** or higher with **NPM** version **10.7.0** or higher ( **`brew install node`** ).

Note that running outside docker is optional. Moreover, when running everything, outside docker, you should see more or less a 2x performance increase.

-->

The requirements in terms of hardware and OS are:

* **MacBook Pro** with **M1** or **M2** cpu ( no other machines have been used for testing )
* **MacOs** updated to **Sonoma 14.5** or higher ( as no other operating systems have been used for testing )

The software requirements are:

* **Docker**: version 25.0.2 or higher ( **`brew cask install docker`** )
* **Docker Compose**: version v2.24.3-desktop.1 or higher ( **`sudo -s` followed by `curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose`** )

The requirements for running **without virtualisation** are:

* **Python**: version 3.10.14 or higher ( **`brew install python@3.10`** )
	+ **PIP**: version 24.1 or higher

Note: Running _outside_ Docker is **optional**, therefore a minimum dependency list is required, however, if an increased performance is needed, one can run this demo without docker, meaning the other dependencies will need to be installed ( if a CUDA graphic card is present then the training time will be decreased significantly ).

---

## ZKML MNIST Local Demo: Detailing the Full Setup

This repository is centered around two Python-based Docker virtual machines ( aka containers ), either of which can be used for training and export a simple model ( the 1st one, **`modellerjupyter`**, could be used when giving the demo to a more academic audience whereas the 2nd one, **`modellerauto`** could be used when giving the demo to a more traditional tech-savvy audience ).

That being said, the two previously mentioned Python pieces of code are containerised using **Docker** and orchestrated using **Docker Compose** in an all-encompassing YAML file ( **`docker-compose.yaml`** ). This is highlighted in the table below:

| Service              	| Description                                                                                                 	| Port   	| Link                                        	|
|----------------------	|-------------------------------------------------------------------------------------------------------------	|--------	|---------------------------------------------	|
| `zk-modellerjupyter` 	| A Jupyter notebook-based model development and training environment for building the MNIST ML model.        	| `8888` 	| **[localhost:8888](http://localhost:8888)** 	|
| `zk-modellerauto`    	| An auto-pilot-like Python environment that automates the process of training and exporting MNIST ML models. 	| `8081` 	| **[localhost:8081](http://localhost:8081)** 	|

---

## ZKML MNIST Local Demo: How To Start The Modeller Demo

To start the demo, follow these steps:

```sh
docker-compose up -d
```

Note that you can add a service name at the end of the above ( **`docker-compose up -d zk-modellerjupyter`** or **`docker-compose up -d zk-modellerauto`** ), otherwise you will be spinning up all services. Also, this doesn't include building / re-building the images therefore, to do that use: **`docker-compose up --build -d`**.

To monitor the logs of the application, run: **`docker-compose logs zk-modellerjupyter zk-modellerauto --tail 10`**.

This command will show you the last 10 lines of the logs for each service. You can adjust the **`--tail`** parameter accordingly or even add it to watch with: **`watch -ct docker-compose logs zk-modellerjupyter zk-modellerauto --tail 10`**.

Last, but not least, to stop and remove all containers, run **`docker-compose down --remove-orphans -v`** or in order to jump inside one of the containers, use: **`docker exec -it zk-modellerjupyter sh`** OR **`docker exec -it zk-modellerauto sh`**.

---

## ZKML MNIST Local Demo: System Architecture Diagram

To get a good grasp at first glance, or at least a satisfactory one, as to what this project entails, please review the diagram.

```
┌────────────────────────────────────┐    ┌───────────────────────────┐
│                                    │    │                           │
│ Name:zk-modellerjupyter Port:8888  │ ─► │ Host-guest shared         │
│                                    │    │                           │
│ ┌────────────────────────────────┐ │ ◄─ │ files (optional):         │
│ │                                │ │    │                           │
│ │ JupyterNotebook                │ │    │ mnist-train-n-test.tar.gz │
│ │                                │ │    │                           │
│ │ Renders: mnist.ipynb           │ │    │ models-pkl.tar.gz         │
│ │                                │ │    │                           │
│ │ ( from ./jupyter/ )            │ │    │                           │
│ │                                │ │    │                           │
│ │ Outputs: models AND cal_data   │ │    │                           │
│ │                                │ │    │                           │
│ └────────────────────────────────┘ │    │                           │
│                                    │    │                           │
└────────────────────────────────────┘    └───────────────────────────┘

 │
 ▼
 Independent services ( not connected )
 ▲
 │

┌────────────────────────────────────┐    ┌───────────────────────────┐
│                                    │    │                           │
│ Name:zk-modellerauto Port:8081     │ ─► │ Host-guest shared         │
│                                    │    │                           │
│ ┌────────────────────────────────┐ │ ◄─ │ files (optional):         │
│ │                                │ │    │                           │
│ │ AutoNotebook                   │ │    │ mnist-train-n-test.tar.gz │
│ │                                │ │    │                           │
│ │ Renders: mnist.py from ./app/  │ │    │ models-pkl.tar.gz         │
│ │                                │ │    │                           │
│ │ ServesOutFiles: localhost:8081 │ │    │                           │
│ │                                │ │    │                           │
│ │ Outputs: models AND cal_data   │ │    │                           │
│ │                                │ │    │                           │
│ └────────────────────────────────┘ │    │                           │
│                                    │    │                           │
└────────────────────────────────────┘    └───────────────────────────┘
```

**Note**: To make the Auto / Jupyter modeller download the mnist dataset again, the first Tar Gzip file must be removed / deactivated ( **`mnist-train-n-test.tar.gz`** ).

**Note 2**: To make the Auto / Jupyter modeller re-do all the training steps, the second Tar Gzip file must be removed / deactivated ( **`models-pkl.tar.gz`** ), otherwise the modeller will simply act as an **ONNX** and **Calibration Data** exporter.

---

---
