# Hyper Parameters in Experiments

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
| LeNet  | 1e-3 | 0.9 | [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60] | 40 |
| VFNet-A\*\* | 1e-3 | 0.9 | [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60] | 60 |
| MLP-Mixer | 1e-3 | 0.9 | [20, 40, 60] | 25 |
| VIT-2/8\* | 1e-3 | 0.9 | [5, 10, 20, 30, 40, 55, 60, 70, ] | 75 |
| VFNet-B\*\* | 1e-3 | 0.9 | [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60] | 60 |
| MLP-3  | 1e-3 | 0.9 | [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60] | 60 |
| MLP-2  | 1e-3 | 0.9 | [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60] | 60 |
| Softmax regression | 1e-3 | 0.9 | [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60] | 60 |

\* VIT-2/p means a ViT with 2 encoder blocks and patch_size=p.
\*\* VFNet has multi-structures. A: FBlock after 2nd conv. B: FBlock after 1st conv. Then canceled LeNet's 1st fc layer.
