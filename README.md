# MNIST-pytorch

Train various models on MNIST and other datasets! 、
We will select models and datasets as small as possible so that these models can be easily trained on a laptop's GPU.

## Commands

Usage:

```sh
python main.py <command> -m <model-type> -d <dataset-name> -n <model-name>
```

A command could be: train, test, summary, show
A dataset_name could be: mnist, cifar-10
A model_type could be: lenet, mlp1, mlp2, mlp3

Example:

```sh
python main.py train -d cifar-10 -m lenet
```

## Requirements

+ torch==1.1.0
+ torchsummary=1.5.1
+ torchvision=0.4.1
+ einops==0.3.0
+ matplotlib=3.3.3
+ numpy=1.19.3
+ tqdm=4.54.1

## Files

+ src/: Source codes.
+ data/: Data, include mnist and cifar-10 by now.
+ saved/: Saved model parameters.

## Results
