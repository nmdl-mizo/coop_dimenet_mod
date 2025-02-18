# -*- coding: utf-8 -*-
# Description: This file contains the definition of the COOP_DimeNet_mod model.
from typing import Callable, Union, Optional as OptTensor
import torch
from torch import Tensor
from torch_geometric.nn.models import DimeNetPlusPlus
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.nn import radius_graph
from torch_geometric.nn.models.dimenet import InteractionPPBlock, OutputPPBlock, triplets
import numpy as np
from scipy.special import factorial

np.math = type('math', (), {})()
np.math.factorial = factorial


class COOP_DimeNet_mod(DimeNetPlusPlus):
    def __init__(
        self,
        hidden_channels: int = 128,
        out_channels: int = 256,
        num_blocks: int = 4,
        int_emb_size: int = 64,
        basis_emb_size: int = 8,
        out_emb_channels: int = 256,
        num_spherical: int = 7,
        num_radial: int = 6,
        coop_channels: int = 201,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 1,
        num_output_layers: int = 3,
        act: Union[str, Callable] = 'swish',
    ):
        act = activation_resolver(act)
        super().__init__(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            int_emb_size=int_emb_size,
            basis_emb_size=basis_emb_size,
            out_emb_channels=out_emb_channels,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
            act=act,
        )

        # We are re-using the RBF, SBF and embedding layers of `DimeNet` and
        # redefine output_block and interaction_block in DimeNet++.
        # Hence, it is to be noted that in the above initalization, the
        # variable `num_bilinear` does not have any purpose as it is used
        # solely in the `OutputBlock` of DimeNet:
        self.output_blocks = torch.nn.ModuleList([
            OutputPPBlock(
                num_radial,
                hidden_channels,
                out_emb_channels,
                out_channels,
                num_output_layers,
                act,
            ) for _ in range(num_blocks + 1)
        ])

        self.interaction_blocks = torch.nn.ModuleList([
            InteractionPPBlock(
                hidden_channels,
                int_emb_size,
                basis_emb_size,
                num_spherical,
                num_radial,
                num_before_skip,
                num_after_skip,
                act,
            ) for _ in range(num_blocks)
        ])

        self.coop_output_blocks = torch.nn.ModuleList([
            torch.nn.Linear(out_channels, hidden_channels),  # first hidden layer
            torch.nn.ReLU(),  # ReLU activation
            torch.nn.Linear(hidden_channels, coop_channels)  # second hidden layer to output
        ])

        self.reset_parameters()


    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        edgelist_forbatch: Tensor,
        batch: OptTensor = None,
    ) -> Tensor:
        r"""
        Args:
            z (torch.Tensor): Atomic number of each atom with shape
                :obj:`[num_atoms]`.
            pos (torch.Tensor): Coordinates of each atom with shape
                :obj:`[num_atoms, 3]`.
            batch (torch.Tensor, optional): Batch indices assigning each atom
                to a separate molecule with shape :obj:`[num_atoms]`.
                (default: :obj:`None`)
        """
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            edge_index, num_nodes=z.size(0))

        # Calculate distances.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        pos_i = pos[idx_i]
        pos_ji, pos_ki = pos[idx_j] - pos_i, pos[idx_k] - pos_i
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki, dim=-1).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block.
        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        # Interaction blocks.
        for interaction_block, output_block in zip(self.interaction_blocks,
                                                   self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P = P + output_block(x, rbf, i, num_nodes=pos.size(0))

        if batch is None:
            return P.sum(dim=0)
        else:
            result = []
            for i in range(edgelist_forbatch.shape[0]):
                result.append(P[edgelist_forbatch[i][0]] + P[edgelist_forbatch[i][1]])
            result = torch.stack(result)
            for coop_output_block in self.coop_output_blocks:
                result = coop_output_block(result)
            return result
