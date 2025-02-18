# COOP_DimeNet_mod

Graph neural network model for predicting overlap population based on DimeNet++

## Overview

This provides the code and weights of a graph neural network for predicting overlap populations (OPs) of molecules.
The model is developed based on DimeNet++[1] implementation in PyTorch Geometric[2].
A weight was trained on OPs of organic molecules in QM9 dataset[3].
The dataset of OPs was calculated by aims_chembond[4] implemented in FHI-aims[5] and is separately published at Zenodo[6].

## Model architecture

This model is based on the code of DimeNet++ in PyTorch Geometric and creates edge features from the node features of the two sites that comprise the bond to be predicted for the OP, and performs a nonlinear transformation with multilayer perceptron to regress the OP's spectral regression of the data.

The detailed description will be presented in a paper[7].

## Environment setup

This repository contains files for making a python environment for running COOP_DimeNet_mod using uv.
1. Install uv following [this page](https://docs.astral.sh/uv/getting-started/installation/).
1. Colone this repository.
1. run uv in the repository directory as follows:
   ```bash
   uv sync
   ```

## Dependencies

The project relies on the following main dependencies:
- PyTorch
- PyTorch Geometric
- NumPy
- SciPy

For a complete list of dependencies, please refer to `pyproject.toml` and `uv.lock`.

## Usage

The model for QM9 can be loaded as follows:
```python
import torch
from coop_dimenet_mod import COOP_DimeNet_mod

# Load the model
model = COOP_DimeNet_mod(
    hidden_channels=128,
    out_channels=256,
    num_blocks=4,
    int_emb_size=64,
    basis_emb_size=8,
    out_emb_channels=256,
    num_spherical=7,
    num_radial=6,
    cutoff=5.0,
    coop_channels=201,
    num_after_skip=1,
).to('cuda')
model.load_state_dict(torch.load("weights/checkpoint.pt", weights_only=True))
```

The overlap populations of molecules can be predicted as follows:
```python
op = model.forward(
    pos=data.pos,
    z=data.z,
    edgelist_forbatch=target_edge_index.T,
    batch=data.batch
)
```
where `data` is `torch_geometric.data.Batch` object containing molecular graphs and `target_edge_index` is an array describing bonds of interest with shape of `[number of bonds of interest, 2]`.
The output is the predicted OPs with shape of `[number of bonds of interest, coop_channels]`. 

In case of the trained model, the coop_channels is 201 ranging from -30 eV to 0 eV relative to vacuum level calculated in FHI-aims.

An example for prediction and plotting OPs of benzene and fluorobenzen is provided in `example/test_CC.ipynb`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

1. J. Gasteiger, S. Giri, J. T. Margraf, S. Günnemann, [arXiv:2011.14115](https://arxiv.org/abs/2011.14115) (2020)
1. L. Ruddigkeit, R. van Deursen, L. C. Blum, and J.-L. Reymond, J. Chem. Inf. Model. 52, 2864–2875 (2012); R. Ramakrishnan, P. O. Dral, M. Rupp, and O. A. von Lilienfeld, Sci. Data 1, 140022 (2014)
1. M. Fey, and J. E. Lenssen, ICLR Workshop on Representation Learning on Graphs and Manifolds (2019)
1. V. Blum et al., Comp. Phys. Commun. 180, 2175-2196 (2009)
1. I. Takahara, K. Shibata, and T. Mizoguchi, Modelling Simul. Mater. Sci. Eng. 32, 055028 (2024)
1. N. Riuki, I. Takahara, K. Shibata, and T. Mizoguchi, Zenodo (2025) http://doi.org/10.5281/zenodo.14695630
1. N. Riuki, I. Takahara, K. Shibata, and T. Mizoguchi, submitted (2025)
