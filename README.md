# CIFAR10 Playground

## Mixup 
- 200 epochs, cut down learning rate by 10 at epoch 100 and 150.

|Method|Accuracy|Config|Notes|
|:----:|:-----:|:-----:|:---:|
|Baseline|94.86|vanilla\_200.yaml||
|Mixup|96.01|-|$\alpha$=1|
|Manifold Mixup|96.10|-|$\alpha$=1 <br> layers=[0,1]|
|Manifold Mixup|96.01|-|$\alpha$=1 <br> layers=[0,1,2]|

- 1200 epochs, cut down learning rate by 10 at epoch 400 and 800.

|Method|Accuracy|Config|Notes|
|:----:|:-----:|:-----:|:---:|
|Baseline|95.59|vanilla\_1200.yaml||
|Mixup|96.85|mixup.yaml|$\alpha$=1|
|Manifold Mixup|97.19|manifold\_mixup.yaml|$\alpha$=1 <br> layers=[0,1,2]|

## Training with GAN Data
- Images generated from conditional BigGAN.
- 200 epochs, cut down learning rate by 10 at 100 and 150.

|\% Generated Data|Accuracy|Config|
|:----:|:-----:|:-----:|
|\%100|66.59|gan\_100.yaml|
|\%50|92.34|gan\_50.yaml|
|\%25|93.67|gan\_25.yaml|
|\%10|94.33|gan\_10.yaml|
|\%0|94.86|vanilla\_200.yaml||

## Augmentations
- 200 epochs, cut down learning rate by 10 at 100 and 150.

|Method|Accuracy|Config|
|:----:|:-----:|:-----:|
|Baseline|94.86|vanilla\_200.yaml|
|Cutout|-|cutout.yaml|
|AutoAugment|-|autoaugment.yaml|
|AutoAugment+Cutdown|-|cutout\_autoaugment.yaml|
|RandAugment|-|randaugment.yaml||
