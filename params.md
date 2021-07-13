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
| LANet\*\*\* | 1e-3 | 0.9 | [0, 10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145] | 150 |
| MLP-Mixer | 1e-3 | 0.9 | [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60] | 60 |
| VIT-2/8\* | 1e-3 | 0.9 | [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60] | 60 |
| VFNet-B\*\* | 1e-3 | 0.9 | [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60] | 60 |
| MLP-3  | 1e-3 | 0.9 | [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60] | 60 |
| MLP-2  | 1e-3 | 0.9 | [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60] | 60 |
| Softmax regression | 1e-3 | 0.9 | [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60] | 60 |

\* VIT-2/p means a ViT with 2 encoder blocks and patch_size=p.
\*\* VFNet has multi-structures. A: FBlock after 2nd conv. B: FBlock after 1st conv. Then canceled LeNet's 1st fc layer.
\*\*\* lanet_20210713_122503_150
