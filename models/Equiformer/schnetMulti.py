import os
import os.path as osp
import warnings
from math import pi as PI
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from torch.nn import Embedding, Linear, ModuleList, Sequential, ReLU

from torch_geometric.data import Dataset, download_url, extract_zip
from torch_geometric.io import fs
from torch_geometric.nn import MessagePassing, SumAggregation, radius_graph
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
from torch_geometric.typing import OptTensor
from torch_geometric.nn import global_add_pool, global_mean_pool

qm9_target_dict: Dict[int, str] = {
    0: 'dipole_moment',
    1: 'isotropic_polarizability',
    2: 'homo',
    3: 'lumo',
    4: 'gap',
    5: 'electronic_spatial_extent',
    6: 'zpve',
    7: 'energy_U0',
    8: 'energy_U',
    9: 'enthalpy_H',
    10: 'free_energy',
    11: 'heat_capacity',
}


class SchNet(torch.nn.Module):
    r"""The continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_ paper that uses
    the interactions blocks of the form.

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    here :math:`h_{\mathbf{\Theta}}` denotes an MLP and
    :math:`\mathbf{e}_{j,i}` denotes the interatomic distances between atoms.

    .. note::

        For an example of using a pretrained SchNet variant, see
        `examples/qm9_pretrained_schnet.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        qm9_pretrained_schnet.py>`_.

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_filters (int, optional): The number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): The number of interaction blocks.
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        interaction_graph (callable, optional): The function used to compute
            the pairwise interaction graph and interatomic distances. If set to
            :obj:`None`, will construct a graph based on :obj:`cutoff` and
            :obj:`max_num_neighbors` properties.
            If provided, this method takes in :obj:`pos` and :obj:`batch`
            tensors and should return :obj:`(edge_index, edge_weight)` tensors.
            (default :obj:`None`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        readout (str, optional): Whether to apply :obj:`"add"` or :obj:`"mean"`
            global aggregation. (default: :obj:`"add"`)
        dipole (bool, optional): If set to :obj:`True`, will use the magnitude
            of the dipole moment to make the final prediction, *e.g.*, for
            target 0 of :class:`torch_geometric.datasets.QM9`.
            (default: :obj:`False`)
        mean (float, optional): The mean of the property to predict.
            (default: :obj:`None`)
        std (float, optional): The standard deviation of the property to
            predict. (default: :obj:`None`)
        atomref (torch.Tensor, optional): The reference of single-atom
            properties.
            Expects a vector of shape :obj:`(max_atomic_number, )`.
    """

    url = 'http://www.quantum-machine.org/datasets/trained_schnet_models.zip'

    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        interaction_graph: Optional[Callable] = None,
        max_num_neighbors: int = 32,
        readout: str = 'add',
        dipole: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        atomref: OptTensor = None,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.dipole = dipole
        self.sum_aggr = SumAggregation()
        self.readout = aggr_resolver('sum' if self.dipole else readout)
        self.mean = mean
        self.std = std
        self.scale = None

        if self.dipole:
            import ase

            atomic_mass = torch.from_numpy(ase.data.atomic_masses)
            self.register_buffer('atomic_mass', atomic_mass)

        # Support z == 0 for padding atoms so that their embedding vectors
        # are zeroed and do not receive any gradients.
        self.embedding = Embedding(100, hidden_channels, padding_idx=0)

        if interaction_graph is not None:
            self.interaction_graph = interaction_graph
        else:
            self.interaction_graph = RadiusInteractionGraph(
                cutoff, max_num_neighbors)

        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            self.interactions.append(block)

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, 1)

        self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = Embedding(100, 1)
            self.atomref.weight.data.copy_(atomref)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        if self.atomref is not None:
            self.atomref.weight.data.copy_(self.initial_atomref)

    @staticmethod
    def from_qm9_pretrained(
        root: str,
        dataset: Dataset,
        target: int,
    ) -> Tuple['SchNet', Dataset, Dataset, Dataset]:  # pragma: no cover
        r"""Returns a pre-trained :class:`SchNet` model on the
        :class:`~torch_geometric.datasets.QM9` dataset, trained on the
        specified target :obj:`target`.
        """
        import ase
        import schnetpack as spk  # noqa

        assert target >= 0 and target <= 12
        is_dipole = target == 0

        units = [1] * 12
        units[0] = ase.units.Debye
        units[1] = ase.units.Bohr**3
        units[5] = ase.units.Bohr**2

        root = osp.expanduser(osp.normpath(root))
        os.makedirs(root, exist_ok=True)
        folder = 'trained_schnet_models'
        if not osp.exists(osp.join(root, folder)):
            path = download_url(SchNet.url, root)
            extract_zip(path, root)
            os.unlink(path)

        name = f'qm9_{qm9_target_dict[target]}'
        path = osp.join(root, 'trained_schnet_models', name, 'split.npz')

        split = np.load(path)
        train_idx = split['train_idx']
        val_idx = split['val_idx']
        test_idx = split['test_idx']

        # Filter the splits to only contain characterized molecules.
        idx = dataset.data.idx
        assoc = idx.new_empty(idx.max().item() + 1)
        assoc[idx] = torch.arange(idx.size(0))

        train_idx = assoc[train_idx[np.isin(train_idx, idx)]]
        val_idx = assoc[val_idx[np.isin(val_idx, idx)]]
        test_idx = assoc[test_idx[np.isin(test_idx, idx)]]

        path = osp.join(root, 'trained_schnet_models', name, 'best_model')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            state = fs.torch_load(path, map_location='cpu')

        net = SchNet(
            hidden_channels=128,
            num_filters=128,
            num_interactions=6,
            num_gaussians=50,
            cutoff=10.0,
            dipole=is_dipole,
            atomref=dataset.atomref(target),
        )

        net.embedding.weight = state.representation.embedding.weight

        for int1, int2 in zip(state.representation.interactions,
                              net.interactions):
            int2.mlp[0].weight = int1.filter_network[0].weight
            int2.mlp[0].bias = int1.filter_network[0].bias
            int2.mlp[2].weight = int1.filter_network[1].weight
            int2.mlp[2].bias = int1.filter_network[1].bias
            int2.lin.weight = int1.dense.weight
            int2.lin.bias = int1.dense.bias

            int2.conv.lin1.weight = int1.cfconv.in2f.weight
            int2.conv.lin2.weight = int1.cfconv.f2out.weight
            int2.conv.lin2.bias = int1.cfconv.f2out.bias

        net.lin1.weight = state.output_modules[0].out_net[1].out_net[0].weight
        net.lin1.bias = state.output_modules[0].out_net[1].out_net[0].bias
        net.lin2.weight = state.output_modules[0].out_net[1].out_net[1].weight
        net.lin2.bias = state.output_modules[0].out_net[1].out_net[1].bias

        mean = state.output_modules[0].atom_pool.average
        net.readout = aggr_resolver('mean' if mean is True else 'add')

        dipole = state.output_modules[0].__class__.__name__ == 'DipoleMoment'
        net.dipole = dipole

        net.mean = state.output_modules[0].standardize.mean.item()
        net.std = state.output_modules[0].standardize.stddev.item()

        if state.output_modules[0].atomref is not None:
            net.atomref.weight = state.output_modules[0].atomref.weight
        else:
            net.atomref = None

        net.scale = 1.0 / units[target]

        return net, (dataset[train_idx], dataset[val_idx], dataset[test_idx])

    def forward(self, z: Tensor, pos: Tensor,
                batch: OptTensor = None) -> Tensor:
        r"""Forward pass.

        Args:
            z (torch.Tensor): Atomic number of each atom with shape
                :obj:`[num_atoms]`.
            pos (torch.Tensor): Coordinates of each atom with shape
                :obj:`[num_atoms, 3]`.
            batch (torch.Tensor, optional): Batch indices assigning each atom
                to a separate molecule with shape :obj:`[num_atoms]`.
                (default: :obj:`None`)
        """
        batch = torch.zeros_like(z) if batch is None else batch

        h = self.embedding(z)
        edge_index, edge_weight = self.interaction_graph(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if self.dipole:
            # Get center of mass.
            mass = self.atomic_mass[z].view(-1, 1)
            M = self.sum_aggr(mass, batch, dim=0)
            c = self.sum_aggr(mass * pos, batch, dim=0) / M
            h = h * (pos - c.index_select(0, batch))

        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(z)

        out = self.readout(h, batch, dim=0)

        if self.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)

        if self.scale is not None:
            out = self.scale * out

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions}, '
                f'num_gaussians={self.num_gaussians}, '
                f'cutoff={self.cutoff})')


# ---------------------
# Cross-Attention Fusion Module
# ---------------------
class CrossAttentionFusion(nn.Module):
    def __init__(self, mol_dim: int, clip_dim: int, attn_dim: int = 256, num_heads: int = 4):
        super().__init__()
        # Project molecular features to the attention space.
        self.mol_proj = Linear(mol_dim, attn_dim)
        # Project clip features to the attention space.
        self.clip_proj = Linear(clip_dim, attn_dim)
        # Multihead attention module. We use batch_first=True so inputs are [B, seq_len, attn_dim].
        self.attn = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=num_heads, batch_first=True)
        # Project the attended output back to the molecular dimension.
        self.out_proj = Linear(attn_dim, mol_dim)

    def forward(self, h: torch.Tensor, clip_feat: torch.Tensor) -> torch.Tensor:
        # h: [B, mol_dim]; clip_feat: [B, clip_dim]
        # Expand both to have a sequence length of 1.
        h_proj = self.mol_proj(h).unsqueeze(1)  # [B, 1, attn_dim]
        clip_proj = self.clip_proj(clip_feat).unsqueeze(1)  # [B, 1, attn_dim]
        # Use clip features as both key and value and h as the query.
        attn_out, _ = self.attn(query=h_proj, key=clip_proj, value=clip_proj)
        attn_out = attn_out.squeeze(1)  # [B, attn_dim]
        fusion = self.out_proj(attn_out)  # [B, mol_dim]
        # Residual connection
        return h + fusion

#
# # ---------------------
# # Updated SchNetMulti with Cross-Modal Attention Fusion
# # ---------------------
# class SchNetMulti(nn.Module):
#     def __init__(
#             self,
#             clip_encoder: int = 32,
#             hidden_channels: int = 128,  # Updated initial embedding dimension.
#             num_filters: int = 128,
#             num_interactions: int = 6,
#             num_gaussians: int = 50,
#             cutoff: float = 10.0,
#             interaction_graph: Optional[Callable] = None,
#             max_num_neighbors: int = 32,
#             readout: str = 'add',
#             dipole: bool = False,
#             mean: Optional[float] = None,
#             std: Optional[float] = None,
#             atomref: OptTensor = None,
#     ):
#         super().__init__()
#         self.clip_encoder = clip_encoder
#         self.hidden_channels = hidden_channels
#         self.num_filters = num_filters
#         self.num_interactions = num_interactions
#         self.num_gaussians = num_gaussians
#         self.cutoff = cutoff
#         self.dipole = dipole
#         self.mean = mean
#         self.std = std
#         self.scale = None
#
#         # Set the aggregation function based on the readout type.
#         if readout == 'add':
#             self.readout = global_add_pool
#         elif readout == 'mean':
#             self.readout = global_mean_pool
#         else:
#             raise ValueError(f"Unsupported readout type: {readout}")
#
#         # Embedding layer for atomic numbers. Now uses 768-dimensional embeddings.
#         self.embedding = Embedding(100, hidden_channels, padding_idx=0)
#
#         if interaction_graph is not None:
#             self.interaction_graph = interaction_graph
#         else:
#             self.interaction_graph = RadiusInteractionGraph(cutoff, max_num_neighbors)
#
#         # Gaussian smearing layer for distance expansion.
#         self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
#
#         # Interaction blocks.
#         self.interactions = ModuleList()
#         for _ in range(num_interactions):
#             # Assume InteractionBlock is defined/imported elsewhere.
#             block = InteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff)
#             self.interactions.append(block)
#
#         # Branch to process 512-dim CLIP embeddings.
#         self.clip_head = Sequential(
#             Linear(768, hidden_channels),
#             ReLU(),
#             Linear(hidden_channels, self.clip_encoder)
#         )
#         # Use cross-attention fusion to integrate molecular and clip features.
#         # After lin_target, the molecular representation will have dimension hidden_channels//2.
#         self.cross_attn_fusion = CrossAttentionFusion(
#             mol_dim=hidden_channels // 2,
#             clip_dim=self.clip_encoder,
#             attn_dim=128,
#             num_heads=4
#         )
#
#         # Layers to process the molecular representation.
#         self.lin_target = Sequential(
#             Linear(hidden_channels, hidden_channels // 2),
#             ReLU(),
#         )
#         self.lin_target_2 = Sequential(
#             Linear(hidden_channels // 2, hidden_channels // 4),
#             ReLU(),
#         )
#         self.lin_target_3 = Sequential(
#             Linear(hidden_channels // 4, 1)
#         )
#         # Registering buffers for atom references.
#         self.register_buffer('initial_atomref', atomref)
#         self.atomref = None
#         if atomref is not None:
#             self.atomref = Embedding(100, 1)
#             self.atomref.weight.data.copy_(atomref)
#
#     def forward(self, z: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None,
#                 clip_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
#         """
#         Forward pass to compute outputs for molecules including the main target and constrained properties.
#
#         Args:
#             z (Tensor): Atomic numbers [num_atoms].
#             pos (Tensor): Atom positions [num_atoms, 3].
#             batch (Tensor, optional): Batch indices assigning each atom to a molecule.
#             clip_embeddings (Tensor, optional): CLIP embeddings for each molecule [batch_size, 512].
#
#         Returns:
#             Tensor: Main target predictions.
#         """
#         batch = torch.zeros_like(z) if batch is None else batch
#
#         # Compute initial atomic embeddings.
#         h = self.embedding(z)
#         edge_index, edge_weight = self.interaction_graph(pos, batch)
#         edge_attr = self.distance_expansion(edge_weight)
#
#         # Apply interaction blocks.
#         for interaction in self.interactions:
#             h = h + interaction(h, edge_index, edge_weight, edge_attr)
#
#         # Aggregate node embeddings to obtain a molecular embedding.
#         h = self.readout(h, batch)
#         h = self.lin_target(h)  # h now has dimension hidden_channels//2
#
#         # Process and fuse CLIP embeddings using cross-attention.
#         if clip_embeddings is not None:
#             B = h.size(0)
#             clip_embeddings = clip_embeddings.view(B, 768)
#             clip_feat = self.clip_head(clip_embeddings)  # [B, clip_encoder]
#             h = self.cross_attn_fusion(h, clip_feat)
#
#         # Further processing and final prediction.
#         h = self.lin_target_2(h)
#         target_out = self.lin_target_3(h)
#         return target_out
# --- Gated Fusion Module ---
class GatedFusion(nn.Module):
    def __init__(self, mol_dim: int, clip_dim: int, fusion_dim: int):
        super().__init__()
        # Compute gate values from the concatenated features.
        self.gate = nn.Sequential(
            nn.Linear(mol_dim + clip_dim, fusion_dim),
            nn.Sigmoid()
        )
        # Learn separate projections for each modality.
        self.fc_mol = nn.Linear(mol_dim, fusion_dim)
        self.fc_clip = nn.Linear(clip_dim, fusion_dim)

    def forward(self, mol: torch.Tensor, clip: torch.Tensor) -> torch.Tensor:
        # mol: [B, mol_dim], clip: [B, clip_dim]
        combined = torch.cat([mol, clip], dim=-1)  # [B, mol_dim + clip_dim]
        g = self.gate(combined)  # [B, fusion_dim] with values in [0, 1]
        mol_proj = self.fc_mol(mol)  # [B, fusion_dim]
        clip_proj = self.fc_clip(clip)  # [B, fusion_dim]
        fused = g * mol_proj + (1 - g) * clip_proj  # Adaptive weighted sum
        return fused


# --- Updated SchNetMulti with Gated Fusion ---
class SchNetMulti(nn.Module):
    def __init__(
            self,
            clip_encoder: int = 32,
            hidden_channels: int = 128,
            num_filters: int = 128,
            num_interactions: int = 6,
            num_gaussians: int = 50,
            cutoff: float = 10.0,
            interaction_graph: Optional[Callable] = None,
            max_num_neighbors: int = 32,
            readout: str = 'add',
            dipole: bool = False,
            mean: Optional[float] = None,
            std: Optional[float] = None,
            atomref: OptTensor = None,
    ):
        super().__init__()
        self.clip_encoder = clip_encoder
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.dipole = dipole
        self.mean = mean
        self.std = std
        self.scale = None

        # Set the aggregation function based on the readout type.
        if readout == 'add':
            self.readout = global_add_pool
        elif readout == 'mean':
            self.readout = global_mean_pool
        else:
            raise ValueError(f"Unsupported readout type: {readout}")

        # Embedding layer for atomic numbers.
        self.embedding = Embedding(100, hidden_channels, padding_idx=0)

        if interaction_graph is not None:
            self.interaction_graph = interaction_graph
        else:
            self.interaction_graph = RadiusInteractionGraph(cutoff, max_num_neighbors)

        # Gaussian smearing layer for distance expansion.
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        # Interaction blocks.
        self.interactions = ModuleList()
        for _ in range(num_interactions):
            # Assumes InteractionBlock is defined or imported elsewhere.
            block = InteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff)
            self.interactions.append(block)

        # Branch to process 512-dim CLIP embeddings.
        self.clip_head = Sequential(
            Linear(768, hidden_channels),
            ReLU(),
            Linear(hidden_channels, clip_encoder)
        )
        # Use gated fusion to combine molecular features (dimension = hidden_channels)
        # with processed clip features (dimension = clip_encoder).
        # The fusion output will have dimension equal to hidden_channels.
        self.gated_fusion = GatedFusion(mol_dim=hidden_channels, clip_dim=clip_encoder, fusion_dim=hidden_channels)

        # Layers to further process the fused molecular representation.
        self.lin_target = Sequential(
            Linear(hidden_channels, 1),
        )
        # Register buffers for atom references.
        self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = Embedding(100, 1)
            self.atomref.weight.data.copy_(atomref)

    def forward(self, z: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None,
                clip_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass to compute outputs for molecules including the main target and constrained properties.

        Args:
            z (Tensor): Atomic numbers with shape [num_atoms].
            pos (Tensor): Atom positions with shape [num_atoms, 3].
            batch (Tensor, optional): Batch indices for atoms.
            clip_embeddings (Tensor, optional): CLIP embeddings for each molecule [batch_size, 512].

        Returns:
            Tensor: Main target predictions.
        """
        batch = torch.zeros_like(z) if batch is None else batch

        # Compute initial atomic embeddings.
        h = self.embedding(z)
        edge_index, edge_weight = self.interaction_graph(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)

        # Apply interaction blocks.
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        # Aggregate node embeddings to get a molecular embedding.
        h = self.readout(h, batch)

        # Process and fuse CLIP embeddings using gated fusion, if provided.
        if clip_embeddings is not None:
            B = h.size(0)
            clip_embeddings = clip_embeddings.view(B, 768)
            clip_feat = self.clip_head(clip_embeddings)  # [B, clip_encoder]
            # For gated fusion, we need to match the dimension of h.
            # Option 1: Project h up to hidden_channels.
            # Option 2: Adapt the fusion module.
            # Here we assume h's dimension is intended to be hidden_channels.
            # If h is hidden_channels//2, you could either modify lin_target or adjust fusion.
            # For this example, let's assume h should be projected up:
            h_proj = nn.functional.pad(h, (0, self.hidden_channels - h.size(-1)), mode='constant', value=0)
            h = self.gated_fusion(h_proj, clip_feat)

        # Further processing and final prediction.
        target_out = self.lin_target(h)
        return target_out
class RadiusInteractionGraph(torch.nn.Module):
    r"""Creates edges based on atom positions :obj:`pos` to all points within
    the cutoff distance.

    Args:
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance with the
            default interaction graph method.
            (default: :obj:`32`)
    """
    def __init__(self, cutoff: float = 10.0, max_num_neighbors: int = 32):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    def forward(self, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Forward pass.

        Args:
            pos (Tensor): Coordinates of each atom.
            batch (LongTensor, optional): Batch indices assigning each atom to
                a separate molecule.

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        return edge_index, edge_weight


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels: int, num_gaussians: int,
                 num_filters: int, cutoff: float):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
                           self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                edge_attr: Tensor) -> Tensor:
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class CFConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        nn: Sequential,
        cutoff: float,
    ):
        super().__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                edge_attr: Tensor) -> Tensor:
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W


class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
    ):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.shift

