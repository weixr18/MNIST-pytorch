# Hyper Params in experiments

## MNIST

+ Learning rate: 0.01
+ Optimizer: SGD
+ momentum: 0.9
+ Batch size: 64
+ Epoches: 10
+ Train set size: 50000

## CIFAR

+ Optimizer: Adam
+ Adam betas: (0.9, 0.999)
+ Batch size: 64

| Model | Learning rate | lr decay gamma | lr milestones | Epoches |
| :---: | :---: | :---: | :---: | :---: |
| LeNet  | 1e-3 | 0.9 | [20, 40, 60] | 50 |
| MLP-3  | 1e-3 | 0.9 | [20, 40, 60] | 50 |
| VFNet | 1e-3 | 0.9 | [20, 40, 60] | 50 |
| MLP-Mixer | 1e-3 | 0.9 | [20, 40, 60] | 25 |
| VIT-2  | 1e-3 | 0.9 | [20, 40, 60] | 50 |
| MLP-2  | 1e-3 | 0.9 | [20, 40, 60] | 50 |
| Softmax regression\* | 1e-3 | 0.9 | [20, 40, 60] | 50 |
