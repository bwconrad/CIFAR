# CIFAR10 Sandbox

## Mixup 
### 200 epochs (Learning rate divided by 10 at epoch 100 and 150)

|Method|Accuracy|Config|Notes|
|:----:|:-----:|:-----:|:---:|
|Baseline|94.86|baseline.yaml||
|Mixup|96.01|mixup/200/mixup.yaml|&alpha;=1|
|Manifold Mixup|96.10|mixup/200/manifold\_mixup01.yaml|&alpha;=1 <br> layers=[0,1]|
|Manifold Mixup|96.01|mixup/200/manifold\_mixup012.yaml|&alpha;=1 <br> layers=[0,1,2]|

### 1200 epochs (Learning rate divided by 10 at epoch 400 and 800)

|Method|Accuracy|Config|Notes|
|:----:|:-----:|:-----:|:---:|
|Baseline|95.59|baseline\_1200.yaml||
|Mixup|96.85|mixup/1200/mixup.yaml|&alpha;=1|
|Manifold Mixup|97.19|mixup/manifold\_mixup.yaml|&alpha;=1 <br> layers=[0,1,2]|

## Training with GAN Data
- Images generated from conditional BigGAN.

### 200 epochs (Learning rate divided by 10 at epoch 100 and 150)

|\% Generated Data|Accuracy|Config|
|:----:|:-----:|:-----:|
|100\%|66.23|gan/gan\_100.yaml|
|50\%|92.34|gan/gan\_50.yaml|
|25\%|93.67|gan/gan\_25.yaml|
|10\%|94.33|gan/gan\_10.yaml|
|0\%|94.86|baseline.yaml||

## Augmentations
### 200 epochs (Learning rate divided by 10 at epoch 100 and 150)

|Method|Accuracy|Config|Notes|
|:----:|:-----:|:-----:|:---:|
|Baseline|94.86|baseline.yaml||
|Cutout|95.88|augment/cutout.yaml|cutout=16x16|
|AutoAugment|95.90|augment/autoaugment.yaml||
|AutoAugment + Cutout|96.31|augment/autoaugment\_cutout.yaml|cutout=16x16|
|RandAugment|95.02|augment/randaugment\_n3m5.yaml|n=3 <br> m=5|
|RandAugment|94.93|augment/randaugment\_n3m4.yaml|n=3 <br> m=4|
|RandAugment|94.37|augment/randaugment\_n3m2.yaml|n=3 <br> m=2|
|RandAugment|95.65|augment/randaugment\_n2m5.yaml|n=2 <br> m=5|
|RandAugment|95.50|augment/randaugment\_n2m6.yaml|n=2 <br> m=6|
|RandAugment + Cutout|-|augment/randaugment\_cutout.yaml|n=3 <br> m=5|

- AutoAugment shows better results on CIFAR compared to RandAugment which appears to make the
  augmentations too strong for the size of the model without proper hyperparameter tuning.
  The hyperparameters however offer better flexiblity which should benefit a wider variety
  of datasets compared to AutoAugment which is tuned specifically on a target dataset.
