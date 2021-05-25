# MNIST-pytorch

Train various models on MNIST and other datasets!
\
We will select models and datasets as small as possible so that these models can be easily trained on a laptop's GPU.

## Acknowledgement

Some open-source model impletations are used in this repo:

[https://github.com/rishikksh20/MLP-Mixer-pytorch](https://github.com/rishikksh20/MLP-Mixer-pytorch)
[https://github.com/rishikksh20/FNet-pytorch](https://github.com/rishikksh20/FNet-pytorch)

## Commands

Usage:

```sh
python main.py <command> -m <model-type> -d <dataset-name> -n <model-name>
```

A command could be: train, test, summary, show
\
A dataset_name could be: mnist, cifar-10
\
A model_type could be: lenet, mlp1, mlp2, mlp3, vit, mlpmixer, fnet
\
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

### Conditions

+ Hardwares
  + Intel i7-7550U
  + NVIDIA GeForce MX130

### MNIST

Condition:

+ Learning rate: 0.01
+ Optimizer: SGD
+ momentum: 0.9
+ Batch size: 64
+ Epoches: 10
+ Train set size: 50000

| Model | Accuracy(%) | Train Time(s) | Train Speed(iter/s) | Params |
| :---: | :---: | :---: | :---: | :---: |
| SVM  | **98.23** | 165.13 | -- | -- |
| LeNet  | 97.78 | 50.42 | 155.32 | 21.84k |
| MLP-3  | 97.60 | 28.94 | 270.94 | 251.06k |
| VFNet | 97.32 | 59.27 | 132.28 | 8.49k |
| Random Forest | 97.09 | 310.36 | -- | -- |
| kNN  | 97.08 | **11.74** | -- | -- |
| MLP-Mixer | 96.90 | 127.78 | 61.29  | 20.47k |
| VIT-2  | 96.41 | 291.79 | 26.82 | 217.01k |
| MLP-2  | 96.78 | 23.25 | 336.30 | 39.76k |
| Softmax regression\* | 92.15 | 17.22 | 454.07 | **7.85k** |

\* softmax regression \= MLP-1, or one-layer perceptron.
