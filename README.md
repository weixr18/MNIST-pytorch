# MNIST-pytorch

Train various models on MNIST and other datasets!
\
We will select models and datasets as small as possible so that these models can be easily trained on a laptop's GPU.

## Install

To use LocalAttention module, you need to install it.

```sh
cd src/local_attention
python setup.py install
```

## Usage

```sh
python main.py <command> -m <model-type> -d <dataset-name> -n <model-name>
```

A command could be: train, test, summary, show
\
A dataset_name could be: mnist, cifar-10
\
A model_type could be: lenet, mlp1, mlp2, mlp3, vit, mlpmixer, vfneta, vfnetb, lanet.

\
Example:

```sh
python main.py train -d cifar-10 -m lenet
```

## Requirements

+ torch==1.7.1
+ torchsummary=1.5.1
+ torchvision=0.8.2
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

Cinfigurations:

+ Learning rate: 0.01
+ Optimizer: SGD
+ momentum: 0.9
+ Batch size: 64
+ Epoches: 10
+ Train set size: 50000

| Model | Accuracy(%) | Train Time(s) | Train Speed(iter/s) | Parameter size |
| :---: | :---: | :---: | :---: | :---: |
| SVM | **98.23** | 165.13 | -- | -- |
| LeNet-5 | 97.78 | 50.42 | 155.32 | 21.84k |
| MLP-3  | 97.60 | 28.94 | 270.94 | 251.06k |
| VFNet-2 | 97.32 | 59.27 | 132.28 | 8.49k |
| Random Forest-1000 | 97.09 | 310.36 | -- | -- |
| kNN | 97.08 | **11.74**(val) | -- | -- |
| MLP-Mixer-2 | 96.90 | 127.78 | 61.29  | 20.47k |
| VIT-2  | 96.41 | 291.79 | 26.82 | 217.01k |
| MLP-2  | 96.78 | 23.25 | 336.30 | 39.76k |
| LANet-5/3\*\* | 94.50 | 227.34 | 24.46 | 43.61k |
| Softmax regression\* | 92.15 | 17.22 | 454.07 | **7.85k** |

\* softmax regression \= MLP-1, or one-layer perceptron.
\*\* epoches = 50

Comment: Note the line for knn(k-nearest neighbor). One thing I have to point out is that the hyperparameter used in this result is k=1 and the distance is the cosine distance. This means: simply comparing the cosine distance of the test sample to each training sample and then outputting the label of the nearest training sample achieves an accuracy of 97.01% on MNIST ...... The fact shows that this dataset is really too simple and these results are in fact not very informative.

### CIFAR-10

Hyper parameters and training configurations are [here](./params.md)

| Model | Accuracy(%) | Train Time(s) | Train Speed(iter/s) | Parameter Size |
| :---: | :---: | :---: | :---: | :---: |
| LeNet-5 | **64.68** | 385.47 | 81.07 | 31.34k |
| VFNet-A | 64.15 | 432.58 | 72.24 | 10.79k |
| LANet-3/5 | 59.36 | 2321.12 | 20.68 | 66.23k |
| MLP-Mixer-2 | 59.18 | 706.37 | 44.24 | 34.53k |
| VIT-2/8\* | 56.69 | 1928.27 | 24.86 | 221.8k |
| VFNet-B | 56.22 | 416.96 | 77.16 | 10.79k |
| MLP-3 | 50.60 | 368.55 | 84.79 | 937.5k |
| MLP-2 | 47.04 | 132.79 | 235.32 | 154.1k |
| Random Forest-100 | 42.22 | 89.59 | -- | -- |
| Softmax regression | 39.48 | 88.60 | 352.68 | 30.73k |
| kNN | 35.78 | 16.52 | -- | -- |
| Naive Bayes | 29.64 | 3.69 | -- | -- |

\* VIT-2/p means a ViT with 2 encoder blocks and patch_size=p.

## Acknowledgement

Use some code implementations from these repos:

[https://github.com/rishikksh20/MLP-Mixer-pytorch](https://github.com/rishikksh20/MLP-Mixer-pytorch)

[https://github.com/rishikksh20/FNet-pytorch](https://github.com/rishikksh20/FNet-pytorch)
