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
| MLP-Mixer | 1e-3 | 0.9 | [20, 40, 60] | 25 |
| VIT-2/8\* | 1e-3 | 0.9 | [5, 10, 20, 30, 40, 55, 60, 70, ] | 75 |
| LeNet  | 1e-3 | 0.9 | [20, 40, 60] | 50 |
| VFNet-A\*\* | 1e-3 | 0.9 | [20, 40, 60] | 50 |
| VFNet-B\*\* | 1e-3 | 0.9 | [10, 20, 30, 40, 50] | 50 |
| MLP-3  | 1e-3 | 0.9 | [20, 40, 60] | 50 |
| MLP-2  | 1e-3 | 0.9 | [20, 40, 60] | 50 |
| Softmax regression | 1e-3 | 0.9 | [20, 40, 60] | 50 |
| VIT-2/4\* | 1e-3 | 0.9 | [20, 40, 60] | 50 |

\* VIT-2/p means a ViT with 2 encoder blocks and patch_size=p.
\*\* VFNet has multi-structures. A: FBlock after 2nd conv. B: FBlock after 1st and 2nd conv.
