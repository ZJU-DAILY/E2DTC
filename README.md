# E2​DTC

This repository contains the code used in our paper: E2DTC: An End to End Deep Trajectory ClusteringFramework via Self-Training

## Requirements

- Ubuntu OS
- Python >= 3.5 (Anaconda3 is recommended)
- PyTorch 1.4+
- A Nvidia GPU with cuda 10.2+

Please refer to the source code to install all required packages in Python.

## Data

* Our geolife trajectory clustering datasets are stored in `data` according to our **Ground Truth Generation algorithm**.
* We provide cluster center data, raw trajectory data, as well as discretized token(cell size 300) for training.

## Train

1. Training with parameters

```shell
python e2dtc.py -vocab_size 1319 -knearestvocabs "data/geolife-vocab-dist-cell300.h5" -pretrain_epoch 20 -num_layers 3 -learning_rate 0.0001 -gamma 0.1 -beta 0.1 -update_interval 1 -n_cluster 12 -cuda 0
```
2. The training produces two model `checkpoint.pt` and `best_model.pt`, `checkpoint.pt` contains the latest trained model and `best_model.pt` saves the model which has the best performance on the validation data. 

Some code comes from [t2vec](https://github.com/boathit/t2vec).