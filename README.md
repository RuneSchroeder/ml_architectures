# ML Architectures

A shared PyTorch repository for all machine learning architectures used by the NTLab group


## Installation
First install PyTorch: https://pytorch.org/

Next, install this repository as a package with the following command:

```console
python -m pip install git+https://gitlab.au.dk/tech_ear-eeg/ml_architectures.git@main
```

## Git LFS
Model weights (`.ckpt` files) are stored using [Git LFS](https://git-lfs.com). If you have cloned the repository before the LFS migration, you will need to re-clone to get the correct weight files. Cloning without Git LFS installed will result in LFS pointer files instead of actual weights.

To verify Git LFS is set up correctly after cloning:
```console
git lfs install
git lfs pull
```