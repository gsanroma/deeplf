# DeepLF: Learning deep embeddings for patch-based label fusion

The paper will be referenced here when published

Author: Gerard Sanroma (`gsanroma@gmail.com`)

![GitHub Logo](/images/pipeline.png)

## OS Requirements
anaconda (with python 2.7)

### Python requirements
SimpleITK 0.9.0

Theano 0.8.2

## Method
This method learns transformations of image patches to be used for multi-atlas patch-based label fusion (*Coupe et al. Patch-based Segmentation using Expert Priors: Application to Hippocampus and Ventricle Segmentation. NeuroImage, 54(2): 940–954, 2011*).

It uses a deep neural network to compute non-linear embeddings.

The entry script is `main_train.py`.

There are the following parameters:

### Dataset arguments
- `--train_dir`: directory of training images
- `--val_dir`: directory of validation images
- `--img_suffix`: suffix of images
- `--lab_suffix`: suffix of labelmaps (i.e., annotation masks)

### Sampler arguments
- `--fract_inside`: fraction of patches to sample from the inside of the structure (not border. Set it to default: 0.5)
- `--num_neighbors`: number of neighboring voting patches (default 50)

### storing arguments
- `--model_name`: id of the model (used to name output files)
- `--store_folder`: folder to store model to
- `--label_group`: Group of labels defining a model. A separate model will be learnt from each group of labels. Typically a group of labels is defined as a pair of labels for the left and right parts of a structure

### Training arguments
- `--num_epochs`: number of epochs
- `--train_batch_size`: size of a training batch
- `--est_batch_size`: size of an estimation batch
- `--display_frequency`: frequency (# iterations) for display
- `--segment_frequency`: frequency (# iterations) for segmenting

### optimization arguments
- `--learning_rate`: learning rate
- `--L1_reg`: L1 regularization strength (default 0.0)
- `--L2_reg`: L2 regularization strength (default 0.0)")
- `--sparse_reg`: sparsity regularization strength (default 0.0)
- `--kl_rho`: reference value for KL divergence for sparsity (default 0.05)

### network arguments
- `--num_units`: number of units
- `--num_hidden_layers`: number of hidden layers
- `--activation`: [relu|tanh|sigmoid|none] activation of hidden layers (default relu)
- `--batch_norm`: batch normalization
#### net params from file
- --`load_net`: load net params from file

### method arguments
- `--patch_rad`: image patch radius (default 2)
- `--patch_norm`: patch normalization type [zscore|l2|none]
- `--search_rad`: neighborhood radius for sampling voting patches (default 3)

## Remarks

The method validates periodically the performance by checking the accuracy on batches of patches extracted from the validation images. 
Although this measure is orientative for the progress of training, **it is highly encouraged that validation is performed through actual multi-atlas segmentation experiments on the validation images using some patch-based label fusion method**.

The script `pblf_py.py` does provide an implementation of patch-based label fusion that allows to use a trained neural network to compute patch embeddings.

More details on how to perform such evaluation can be found in the comments of `main_train.py` script.

Various measures showing the progress of training will be periodically plotted into a file (once each `--display_frequency` iterations), including training/validation cost, accuracy and segmentation dices (this latter one in case it is implemented by the user as explained in the code).

## Structure of directories

## Multi-atlas patch-based label fusion script `pblf_py.py`
