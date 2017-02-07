# deeplf
Learning deep embeddings for patch-based label fusion

## OS Requirements
anaconda (with python 2.7)

### Python requirements
SimpleITK 0.9.0
Theano 0.8.2

## Method
This method learns transformations of image patches to be used for patch-based label fusion (Coupe et al. Patch-based Segmentation using Expert Priors: Application to Hippocampus and Ventricle Segmentation. NeuroImage, 54(2): 940–954, 2011).

It uses a deep neural network to compute non-linear embeddings.

The entry script is main\_train.py.
There are the following parameters:

### Dataset arguments
- --train\_dir: directory of training images
- --val\_dir: directory of validation images
- --img\_suffix: suffix of images
- --lab\_suffix: suffix of labelmaps (i.e., annotation masks)

### Sampler arguments
- --fract\_inside: fraction of patches to sample from the inside of the structure (not border. Set it to default: 0.5)
- --num\_neighbors: number of neighboring voting patches (default 50)

### storing arguments
- --model\_name: id of the model (used to name output files)
- --store\_folder: folder to store model to
- --label\_group: Group of labels defining a model. A separate model will be learnt from each group of labels. Typically a group of labels is defined as a pair of labels for the left and right parts of a structure

### Training arguments
- --num\_epochs: number of epochs
- --train\_batch\_size: size of a training batch
- --est\_batch\_size: size of an estimation batch
- --display\_frequency: frequency (# iterations) for display
- --segment\_frequency: frequency (# iterations) for segmenting

### optimization arguments
- --learning\_rate: learning rate
- --L1\_reg: L1 regularization strength (default 0.0)
- --L2\_reg: L2 regularization strength (default 0.0)")
- --sparse\_reg: sparsity regularization strength (default 0.0)
- --kl\_rho: reference value for KL divergence for sparsity (default 0.05)

### network arguments
- --num\_units: number of units
- --num\_hidden\_layers: number of hidden layers
- --activation: [relu|tanh|sigmoid|none] activation of hidden layers (default relu)
- --batch\_norm: batch normalization
#### net params from file
- --load\_net: load net params from file

### method arguments
- --patch\_rad: image patch radius (default 2)
- --patch\_norm: patch normalization type [zscore|l2|none]
- --search\_rad: neighborhood radius for sampling voting patches (default 3)

## Remarks

The method validates periodically the performance by checking the accuracy on batches of patches extracted from the validation images. Although this measure is orientative for the progress of training, it is highly encouraged that validation is performed through actual multi-atlas segmentation experiments on the validation images using some patch-based label fusion method.
The script `pblf_py.py` does provide an implementation of patch-based label fusion that allows to use a trained neural network to compute patch embeddings.
More details on how to perform such evaluation can be found in the `main_train.py` script.
Various measures showing the progress of training will be periodically plotted into a file (once each `--display_frequency` iterations), including training/validation cost, accuracy and segmentation dices (this latter one in case it is implemented by the user as explained in the code).

## Patch-based label fusion script `pblf_py.py`
