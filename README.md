# CIFAR10 Sandbox

## Mixup 
### 200 epochs  (Learning rate divided by 10 at epoch 100 and 150)

|Method|Acc/Error (%)|Config|Parameters|
|:----:|:-----:|:-----:|:---:|
|Baseline|94.86/5.14|baseline/baseline.yaml||
|Mixup|96.01/3.99|mixup/200/mixup.yaml|&alpha;=1|[1]|
|Manifold Mixup|96.10/3.90|mixup/200/manifold\_mixup01.yaml|&alpha;=2 <br> layers=[0,1]|[2]|
|Manifold Mixup|96.01/3.99|mixup/200/manifold\_mixup012.yaml|&alpha;=2 <br> layers=[0,1,2]|
|Cutmix|96.20/3.80|mixup/200/cutmix.yaml|&alpha;=1|[3]|
|Manifold Cutmix|96.06/3.94|mixup/200/manifold\_cutmix\_a1.yaml|&alpha;=1 <br> layers=[0,1,2]|
|Manifold Cutmix|95.70/4.30|mixup/200/manifold\_cutmix\_a2.yaml|&alpha;=2 <br> layers=[0,1,2]|

### 600 epochs  (Learning rate divide by 10 at epoch 300 and 450)   

|Method|Acc/Error (%)|Config|Parameters|
|:----:|:-----:|:-----:|:---:|
|Baseline|95.40/4.60|baseline/baseline\_600.yaml||
|Mixup|96.59/3.41|mixup/600/mixup.yaml|&alpha;=1|
|Manifold Mixup|96.86/3.14|mixup/600/manifold\_mixup\_a2.yaml|&alpha;=2 <br> layers=[0,1,2]|
|Manifold Mixup|96.63/3.37|mixup/600/manifold\_mixup\_a1.yaml|&alpha;=1 <br> layers=[0,1,2]|
|Cutmix|96.76/3.24|mixup/600/cutmix.yaml|&alpha;=1|
|Manifold Cutmix|96.53/3.47|mixup/600/manifold\_cutmix\_a1.yaml|&alpha;=1 <br> layers=[0,1,2]|
|Manifold Cutmix|96.43/3.57|mixup/600/manifold\_cutmix\_a2.yaml|&alpha;=2 <br> layers=[0,1,2]|

### 1200 epochs  (Learning rate divided by 10 at epoch 400 and 800)

|Method|Acc/Error (%)|Config|Parameters|
|:----:|:-----:|:-----:|:---:|
|Baseline|95.59/4.41|baseline/baseline\_1200.yaml||
|Mixup|96.85/3.15|mixup/1200/mixup.yaml|&alpha;=1|
|Manifold Mixup|97.19/2.81|mixup/1200/manifold\_mixup.yaml|&alpha;=2 <br> layers=[0,1,2]|

## Training with GAN Data
- Images generated from conditional BigGAN.

### 200 epochs  (Learning rate divided by 10 at epoch 100 and 150)

|\% Generated Data|Acc/Error (%)|Config|
|:----:|:-----:|:-----:|
|100\%|66.23/33.77|gan/gan\_100.yaml|
|50\%|92.34/7.66|gan/gan\_50.yaml|
|25\%|93.67/6.33|gan/gan\_25.yaml|
|10\%|94.33/5.67|gan/gan\_10.yaml|
|0\%|94.86/5.14|baseline/baseline.yaml||

## Augmentations
- The standard augmentations are random horizonal flips, random translations and
  normalization. Addition augmentations are applied after the random flip and translation while normalization is always applied last. Cutout is applied after Autoaugment/RandAugment when used together.
### 200 epochs  (Learning rate divided by 10 at epoch 100 and 150)

|Method|Acc/Error (%)|Config|Parameters|
|:----:|:-----:|:-----:|:---:|
|Baseline|94.86/5.14|baseline/baseline.yaml||
|Cutout|95.88/4.12|augment/200/cutout.yaml|cutout=16x16|
|AutoAugment|95.90/4.10|augment/200/autoaugment.yaml||
|AutoAugment + Cutout|96.31/3.69|augment/200/autoaugment\_cutout.yaml|cutout=16x16|
|RandAugment|95.02/4.98|augment/200/randaugment\_n3m5.yaml|n=3 <br> m=5|
|RandAugment|94.93/5.07|augment/200/randaugment\_n3m4.yaml|n=3 <br> m=4|
|RandAugment|94.37/5.63|augment/200/randaugment\_n3m2.yaml|n=3 <br> m=2|
|RandAugment|95.65/4.35|augment/200/randaugment\_n2m5.yaml|n=2 <br> m=5|
|RandAugment|95.50/4.50|augment/200/randaugment\_n2m6.yaml|n=2 <br> m=6|
|RandAugment + Cutout|95.64/4.36|augment/200/randaugment\_cutout.yaml|n=2 <br> m=5 <br> cutout=16x16|
|RandAugment (w/ Cutout)|95.63/4.37|augment/200/randaugment\_include\_cutout.yaml|n=2 <br> m=5 |
|GridMask|95.59/4.41|augment/200/gridmask\_8\_32\_r04.yaml|minD=8 <br> maxD=32 <br> r=0.4|
|GridMask|95.88/4.12|augment/200/gridmask\_16\_32\_r04.yaml|minD=16 <br> maxD=32 <br> r=0.4|
|GridMask|95.78/4.22|augment/200/gridmask\_16\_40\_r04.yaml|minD=16 <br> maxD=40 <br> r=0.4|
|GridMask|95.80/4.20|augment/200/gridmask\_16\_32\_r05.yaml|minD=16 <br> maxD=32 <br> r=0.5|
|GridMask|95.69/4.31|augment/200/gridmask\_16\_32\_r03.yaml|minD=16 <br> maxD=32 <br> r=0.3|
|Autoaugment + GridMask|96.18/3.82|augment/200/autoaugment\_gridmask\_16\_32\_r04.yaml|minD=16 <br> maxD=32 <br> r=0.4|
|Autoaugment + GridMask|95.95/4.05|augment/200/autoaugment\_gridmask\_16\_32\_r03.yaml|minD=16 <br> maxD=32 <br> r=0.3|
|AugMix|95.63/4.37|augment/200/augmix\_w3\_d3\_s3.yaml|width=3 <br> depth=3 <br> severity=3|
|AugMix|95.53/4.47|augment/200/augmix\_w3\_d3\_s5.yaml|width=3 <br> depth=3 <br> severity=5|

### 600 epochs  (Learning rate divide by 10 at epoch 300 and 450)   

|Method|Acc/Error (%)|Config|Parameters|
|:----:|:-----:|:-----:|:---:|
|Baseline|95.40/4.60|baseline/baseline\_600.yaml||
|AutoAugment + Cutout|97.09/2.91|augment/600/autoaugment\_cutout.yaml|cutout=16x16|
|RandAugment + Cutout|96.5/3.5|augment/600/randaugment\_cutout.yaml|n=2 <br> m=5 <br> cutout=16x16|
|RandAugment (w/ Cutout)|96.58/3.42|augment/600/randaugment\_include\_cutout.yaml|n=2 <br> m=5|
|Autoaugment + GridMask|97.02/2.98|augment/600/autoaugment\_gridmask\_16\_32\_r04.yaml|minD=16 <br> maxD=32 <br> r=0.4|
|Autoaugment + GridMask|96.81/3.19|augment/600/autoaugment\_gridmask\_16\_32\_r03.yaml|minD=16 <br> maxD=32 <br> r=0.3|
|AugMix|96.33/3.67|augment/600/augmix\_w3\_d3\_s3.yaml|width=3 <br> depth=3 <br> severity=3|


- AutoAugment shows better results on CIFAR compared to RandAugment which appears to make the
  augmentations too strong for the size of the model without proper hyperparameter tuning.
  The hyperparameters however offer better flexiblity which should benefit a wider variety
  of datasets compared to AutoAugment which is tuned specifically on a target dataset.

## Other Regularization 
### 200 epochs  (Learning rate divided by 10 at epoch 100 and 150)

|Method|Acc/Error (%)|Config|Parameters|
|:----:|:-----:|:-----:|:---:|
|Baseline|94.86/5.14|baseline/baseline.yaml||
|Label Smoothing|94.69/5.31|other/smoothing\_01.yaml|&epsilon;=0.1|

## Combinations
### 600 Epochs (Cosine Annealing Schedule)

|Method|Acc/Error (%)|Config|Parameters|
|:----:|:-----:|:-----:|:---:|
|Baseline|95.06/4.94|baseline/baseline\_600\_cos.yaml||
|AutoAugment + Cutout|97.06/2.94|combination/autoaugment\_cutout.yaml|cutout=16x16 |
|AutoAugment + Cutout + Label Smoothing|-|combination/autoaugment\_cutout\_smoothing.yaml|cutout=16x16 <br> &epsilon;=1|
|AutoAugment + Cutout + Mixup|97.25/2.75|combination/autoaugment\_cutout\_mixup.yaml|cutout=16x16 <br> &alpha;=1|
|AutoAugment + Cutout + Mixup + Label Smoothing|97.26/2.74|combination/aa\_cutout\_mixup\_smoothing.yaml|cutout=16x16 <br> &alpha;=1 <br> &epsilon;=0.1|
|AutoAugment + Cutout + Manifold Mixup|97.05/2.95|combination/autoaugment\_cutout\_manifold.yaml|cutout=16x16 <br> &alpha;=2 <br> layers=[0,1,2]|
|AutoAugment + Cutout + Manifold Mixup + Label Smoothing|-|combination/aa\_cutout\_manifold\_smoothing.yaml|cutout=16x16 <br> &alpha;=2 <br> layers=[0,1,2] <br> &epsilon;=0.1|
|AutoAugment + Cutmix|-|combination/autoaugment\_cutmix.yaml|&alpha;=1|



## To Do 
- DropBlock, StochDepth
- Truncated z sampling in GAN
- CIFAR100
- Different architectures
- Architecture modifications  (Shake-Shake, ShakeDrop, etc)
- Further investigate AugMix and GridMask

## References
- [mixup: Beyond Empirical Risk
Minimization](https://arxiv.org/abs/1710.09412)
- [Manifold Mixup: Better Representations by Interpolating Hidden
States](https://arxiv.org/abs/1806.05236)
- [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable
Features](https://arxiv.org/abs/1905.04899)
- [Pretrained BigGAN](https://github.com/ilyakava/BigGAN-PyTorch)
- [Improved Regularization of Convolutional Neural Networks with
  Cutout](https://arxiv.org/abs/1708.04552)
- [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501)
- [RandAugment: Practical automated data augmentation with a reduced search
  space](https://arxiv.org/abs/1909.13719)
- [GridMask Data Augmentation](https://arxiv.org/abs/2001.04086)
- [AugMix: A Simple Data Processing Method to Improve Robustness and
  Uncertainty](https://arxiv.org/abs/1912.02781v2)
