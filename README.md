# Learning Non-Linear Patch Embeddings with Neural Networks for Label Fusion

This repository contains the code for the brain MRI segmentation method described in [this](https://doi.org/10.1016/j.media.2017.11.013) paper.

Author: Gerard Sanroma (`gsanroma@gmail.com`)

![GitHub Logo](/images/pipeline.png)

## Software Requirements
anaconda (with python 2.7)

## License
License for this software package: BSD 3-Clause License. A copy of this license is present in the root directory

### Python requirements
SimpleITK 0.9.0

Theano 0.8.2

## Method
This method learns transformations of image patches to be used for multi-atlas patch-based label fusion.

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
- `--model_name`: id of the model (used to name the output files, which is useful in case of launching several training instances in paralle. More details below)
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
- --`load_net`: alternatively, you can also load an existing net from file

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

**It is also highly encouraged to launch several training scripts in parallel using different values for the hyper parameters.**
The structure of the directories will keep everything organized.
To that end, we should specify all the training instances with the same `--store_folder` and with a different `--model_name`.

## Structure of directories

The method will create the following directory structure under the directory specified in the parameter `store_folder`:

- `Group#`, where `#` is the group number corresponding to the group of labels specified in the parameter `label_group`
- `Labfus`: this directory can be used for evaluating label fusion results on validation images (*Not implemented. To be done by the user*).

- `Group#/grp#_modelname.png`: figure with plots with various measures of training progress (computed once each `--display_frequency` iterations) for model `modelname` and group `#` of labels.
Here, `modelname` is specified through parameter `--model_name`.
- `Group#/modelname/`: directory containing the models stored each `--display_frequency` iterations with format `grp#_epch#.dat`, where `epch#` indicates the epoch number corresponding to the model.

## Multi-atlas patch-based label fusion script `pblf_py.py`

This script features various brands of weighted voting label fusion methods.

The input parameters are the following:

- `--target_img`: to be segmented target image
- `--atlas_img_list`: list of (registered) atlas images
- `--atlas_lab_list`: list of (registered) atlas labelmaps
- `--out_file`: name of output file with fusion result
- `--probabilities`: store segmentation probabilities (experimental)
- `--patch_radius`: image patch radius (default 3x3x3)
- `--search_radius`: search neighborhood radius (default 1x1x1)
- `--fusion_radius`: radius of label patch for fusion (default 1x1x1)
- `--struct_sim`: structural similarity threshold (default 0.9)
- `--normalization`: patch normalization type \[L2 | zscore | none\] (default zscore)
- `--method`: nlwv, nlbeta, deeplf, lasso
  - **nlwv**: non-local weighted voting method as in *Coupe et al. Patch-based Segmentation using Expert Priors: Application to Hippocampus and Ventricle Segmentation. NeuroImage, 54(2): 940–954, 2011*.
  - **nlbeta**: same as **nlwv** but using fixed scale parameter (beta), ideally empirically estimated as proposed in the companion paper of the method (DeepLF).
  - **deeplf**: proposed DeepLF method
  - **lasso**: another variant of similarity weighted voting that computes the weights using sparse linear regression as in *Zhang et al. Sparse patch-based label fusion for multi-atlas segmentation. MBIA 2012*.
- `--regularization`: (nlwv, lasso, nlbeta) regularization parameter for label fusion method (default 0.001)
- `--load_net`: (deeplf) file with the deep neural network
- `--label_grp`: (optional) list of label ids to segment
- `--consensus_thr`: (optional) consensus threshold for creating segmentation mask (default 0.9)
- `--classification_metrics`: (optional) compute classification metrics in non-consensus region (needs target labelmap)

