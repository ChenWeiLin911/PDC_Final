# FourCastNet Project Integration

This repository contains the `FourCastNet` source code used for the project.
The dataset/demo folder and pretrained weights are not included in this repo.
Users must download `ccai_demo` and `FCN_weights_v0` by themselves.

## Expected Directory Layout

After downloading the required external files, the directory should look like:

```text
fourcastnet_project_integration/
├── FourCastNet/          # Included in this repository
│   ├── afno_extension/
│   ├── config/
│   ├── copernicus/
│   ├── data_process/
│   ├── docker/
│   ├── inference/
│   ├── networks/
│   ├── utils/
│   ├── inference.sh
│   ├── library.sh
│   └── train.py
├── ccai_demo/            # Download this separately
│   ├── additional/
│   ├── data/
│   └── model_weights/
└── FCN_weights_v0/       # Download this separately
    └── stats_v0/
```

Only `FourCastNet/` is provided by this repository. Keep `ccai_demo/` and
`FCN_weights_v0/` at the same level as `FourCastNet/` under
`fourcastnet_project_integration/`.

## Environment Setup

Use Anaconda or Miniconda to create a Python 3.10 environment:

```bash
conda create -n fourcastnet python=3.10 -y
conda activate fourcastnet
```

Go to the FourCastNet project directory:

```bash
cd fourcastnet_project_integration/FourCastNet
```

Install the required Python libraries with the provided script:

```bash
./library.sh
```

The script installs PyTorch with CUDA 12.1 support and the Python packages
needed by the project.


## Project Modification

This project adds a new class named `AFNO2DSeparableBMM` in
`FourCastNet/networks/afnonet.py`.

`AFNO2DSeparableBMM` optimizes the AFNO operation by using a separable batched
matrix multiplication implementation. It is used to speed up the AFNO filter
operation compared with the original `AFNO2D` implementation.

To test the AFNO operation speedup, run:

```bash
cd fourcastnet_project_integration/FourCastNet
python test_afno_block_perf.py
```