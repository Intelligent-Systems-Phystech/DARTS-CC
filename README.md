# Neural architecture search with structure complexity control

Authors: [Konstantin Yakovlev](https://github.com/Konstantin-Iakovlev), [Olga Grebenkova](https://github.com/GrebenkovaO), [Oleg Bakhteev](https://github.com/bahleg) and [Vadim Strijov](https://github.com/Strijov).

Contacts: iakovlev.kd(at)phystech.edu

## Annotation
The paper investigates the problem of deep learning model selection. We propose a method of a neural architecture search with respect to the desired model  complexity. An amount of parameters in the model is considered as a model complexity. The proposed method is based on a differential architecture search algorithm (DARTS). Instead of optimizing structural parameters of the architecture, we consider them as a function depending on the complexity parameter. It enables us to obtain multiple architectures at one optimization procedure and select the architecture based on our computation budget.  To evaluate the performance of the proposed algorithm, we conduct experiments on the Fashion-MNIST and CIFAR-10 datasets and compare the resulting architecture with architectures obtained by other neural architecture search  methods.

[TODO: can we publish 1-2 figures from the paper?]

## Technincal details
The core of our NAS implementation is based on the [pt.darts, reimplementation of the DARTS method](https://github.com/khanrc/pt.darts) with some bug fixes (see issues in the original repository).

The main logic of the proposed method can be found at [cnn_darts_hypernet package](models/cnn_darts_hypernet).

We do not use augment.py for the architecture fine-tuning, instead we use search.py with some class changes (see [one_hot_cnn.py](models/cnn/one_hot_cnn.py)) for better model training transparency. The one-host model takes ".json" file with model structure obtained during NAS. We put one-hot non-trainable tensors into the model architecture for the fine-tuning.

All the experiment details are stored into config files, see [configs directory](configs).
## Environment preparation
One can run the experiments in 2 regimes: locally or via Docker. We recommend to use Docker for better reproducibility.

### Local run
To run experiments locally just install packages listed in [requirements.txt](requirements.txt).
```
pip3 install -r requirements.txt
```

### Docker run
To run experiments via Docker you need to build Docker image from the [Dockerfile](Dockerfile).
Note that the docker was configured for the pytorch 1.8.1 distributed for CUDA 11.1. Depending one your hardware, you need to change this dependency in [Dockerfile](Dockerfile). 

To build docker image you can run build_docker.sh:
```
bash build_docker.sh
```

To run the experiments inside docker you should run docker container with opened port for jupyter notebook and shared volume:
```
docker run -p 8888:8888 -v /data/:/nas/searchs -d  --name nas nas-hypernets
docker exec -it nas /bin/bash
```
## Toy experiments

## Large-scale experiment on CIFAR-10
