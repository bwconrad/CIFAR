# CIFAR10 Sandbox

## Mixup 
- 200 epochs (Learning rate divided by 10 at epoch 100 and 150)

|Method|Accuracy|Config|Notes|
|:----:|:-----:|:-----:|:---:|
|Baseline|94.86|vanilla\_200.yaml||
|Mixup|96.01|-|$\alpha$=1|
|Manifold Mixup|96.10|-|$\alpha$=1 <br> layers=[0,1]|
|Manifold Mixup|96.01|-|$\alpha$=1 <br> layers=[0,1,2]|

- 1200 epochs (Learning rate divided by 10 at epoch 400 and 800)

|Method|Accuracy|Config|Notes|
|:----:|:-----:|:-----:|:---:|
|Baseline|95.59|vanilla\_1200.yaml||
|Mixup|96.85|mixup.yaml|$\alpha$=1|
|Manifold Mixup|97.19|manifold\_mixup.yaml|$\alpha$=1 <br> layers=[0,1,2]|

## Training with GAN Data
- Images generated from conditional BigGAN.
- 200 epochs (Learning rate divided by 10 at epoch 100 and 150)

|\% Generated Data|Accuracy|Config|
|:----:|:-----:|:-----:|
|100\%|66.23|gan\_100.yaml|
|50\%|92.34|gan\_50.yaml|
|25\%|93.67|gan\_25.yaml|
|10\%|94.33|gan\_10.yaml|
|0\%|94.86|vanilla\_200.yaml||

## Augmentations
- 200 epochs (Learning rate divided by 10 at epoch 100 and 150)

|Method|Accuracy|Config|Notes|
|:----:|:-----:|:-----:|:---:|
|Baseline|94.86|vanilla\_200.yaml||
|Cutout|95.88|cutout.yaml||
|AutoAugment|95.90|autoaugment.yaml||
|AutoAugment + Cutout|-|autoaugment\_cutout.yaml||
|RandAugment|-|randaugment.yaml|n=3, m=5|
|RandAugment + Cutout|-|randaugment\_cutout.yaml|n=3, m=5|
