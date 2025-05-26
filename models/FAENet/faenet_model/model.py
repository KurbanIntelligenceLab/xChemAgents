"""
FAENet: Frame Averaging Equivariant graph neural Network 
Simple, scalable and expressive model for property prediction on 3D atomic systems.
"""
from typing import Dict, Optional, Union

import torch
from torch import nn
from torch.nn import Embedding, Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.norm import GraphNorm
from torch_scatter import scatter

from .base_model import BaseModel
from .embedding import PhysEmbedding
from .force_decoder import ForceDecoder
from .utils import GaussianSmearing, swish, pbc_preprocess, base_preprocess
from torch_geometric.nn.pool import global_add_pool

class EmbeddingBlock(nn.Module):
    """Initialise atom and edge representations."""

    def __init__(
            self,
            num_gaussians,
            num_filters,
            hidden_channels,
            tag_hidden_channels,
            pg_hidden_channels,
            phys_hidden_channels,
            phys_embeds,
            act,
            second_layer_MLP,
    ):
        super().__init__()
        self.act = act
        self.use_tag = tag_hidden_channels > 0
        self.use_pg = pg_hidden_channels > 0
        self.use_mlp_phys = phys_hidden_channels > 0 and phys_embeds
        self.second_layer_MLP = second_layer_MLP

        # Store dimensions
        self.hidden_channels = hidden_channels
        self.tag_hidden_channels = tag_hidden_channels
        self.pg_hidden_channels = pg_hidden_channels
        self.phys_hidden_channels = phys_hidden_channels

        # Phys embeddings
        self.phys_emb = PhysEmbedding(
            props=phys_embeds, props_grad=phys_hidden_channels > 0, pg=self.use_pg
        )

        # With MLP
        if self.use_mlp_phys:
            self.phys_lin = Linear(self.phys_emb.n_properties, phys_hidden_channels)
        else:
            phys_hidden_channels = self.phys_emb.n_properties

        # Period + group embeddings
        if self.use_pg:
            self.period_embedding = Embedding(
                self.phys_emb.period_size, pg_hidden_channels
            )
            self.group_embedding = Embedding(
                self.phys_emb.group_size, pg_hidden_channels
            )

        # Calculate base embedding size
        self.base_embedding_size = hidden_channels
        if self.use_tag:
            self.base_embedding_size -= tag_hidden_channels
        if phys_hidden_channels > 0:
            self.base_embedding_size -= phys_hidden_channels
        if self.use_pg:
            self.base_embedding_size -= 2 * pg_hidden_channels

        # Main embedding
        self.emb = Embedding(85, self.base_embedding_size)

        # Tag embedding (optional)
        if self.use_tag:
            self.tag_embedding = Embedding(3, tag_hidden_channels)

        # MLP
        self.lin = Linear(hidden_channels, hidden_channels)
        if self.second_layer_MLP:
            self.lin_2 = Linear(hidden_channels, hidden_channels)

        # Edge embedding
        self.lin_e1 = Linear(3, num_filters // 2)  # r_ij
        self.lin_e12 = Linear(num_gaussians, num_filters - (num_filters // 2))  # d_ij
        if self.second_layer_MLP:
            self.lin_e2 = Linear(num_filters, num_filters)

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.reset_parameters()
        if self.use_mlp_phys:
            nn.init.xavier_uniform_(self.phys_lin.weight)
        if self.use_tag:
            self.tag_embedding.reset_parameters()
        if self.use_pg:
            self.period_embedding.reset_parameters()
            self.group_embedding.reset_parameters()
        nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin_e1.weight)
        self.lin_e1.bias.data.fill_(0)
        if self.second_layer_MLP:
            nn.init.xavier_uniform_(self.lin_2.weight)
            self.lin_2.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.lin_e2.weight)
            self.lin_e2.bias.data.fill_(0)

    def forward(self, z, rel_pos, edge_attr, tag=None, subnodes=None):
        # Edge embedding
        rel_pos = self.lin_e1(rel_pos)
        edge_attr = self.lin_e12(edge_attr)
        e = torch.cat((rel_pos, edge_attr), dim=1)
        e = self.act(e)

        if self.second_layer_MLP:
            e = self.act(self.lin_e2(e))

        # Node embedding
        h = self.emb(z)

        # Make sure phys_emb is on the correct device
        if self.phys_emb.device != h.device:
            self.phys_emb = self.phys_emb.to(h.device)

        # Concatenate tag embedding if using tags
        if self.use_tag and tag is not None:
            h_tag = self.tag_embedding(tag)
            h = torch.cat((h, h_tag), dim=1)
        elif self.use_tag:
            # If tag embedding is enabled but no tags provided, add zeros
            h_tag = torch.zeros(h.size(0), self.tag_hidden_channels, device=h.device)
            h = torch.cat((h, h_tag), dim=1)

        # Concatenate physics embeddings
        if self.phys_emb.n_properties > 0:
            h_phys = self.phys_emb.properties[z]
            if self.use_mlp_phys:
                h_phys = self.phys_lin(h_phys)
            h = torch.cat((h, h_phys), dim=1)

        # Concatenate period & group embedding
        if self.use_pg:
            h_period = self.period_embedding(self.phys_emb.period[z])
            h_group = self.group_embedding(self.phys_emb.group[z])
            h = torch.cat((h, h_period, h_group), dim=1)

        # MLP
        h = self.act(self.lin(h))
        if self.second_layer_MLP:
            h = self.act(self.lin_2(h))

        return h, e


class InteractionBlock(MessagePassing):
    """Updates atom representations through custom message passing."""

    def __init__(
        self,
        hidden_channels,
        num_filters,
        act,
        mp_type,
        complex_mp,
        graph_norm,
    ):
        super(InteractionBlock, self).__init__()
        self.act = act
        self.mp_type = mp_type
        self.hidden_channels = hidden_channels
        self.complex_mp = complex_mp
        self.graph_norm = graph_norm
        if graph_norm:
            self.graph_norm = GraphNorm(
                hidden_channels if "updown" not in self.mp_type else num_filters
            )

        if self.mp_type == "simple":
            self.lin_h = nn.Linear(hidden_channels, hidden_channels)

        elif self.mp_type == "updownscale":
            self.lin_geom = nn.Linear(num_filters, num_filters)
            self.lin_down = nn.Linear(hidden_channels, num_filters)
            self.lin_up = nn.Linear(num_filters, hidden_channels)

        elif self.mp_type == "updownscale_base":
            self.lin_geom = nn.Linear(num_filters + 2 * hidden_channels, num_filters)
            self.lin_down = nn.Linear(hidden_channels, num_filters)
            self.lin_up = nn.Linear(num_filters, hidden_channels)

        elif self.mp_type == "updown_local_env":
            self.lin_down = nn.Linear(hidden_channels, num_filters)
            self.lin_geom = nn.Linear(num_filters, num_filters)
            self.lin_up = nn.Linear(2 * num_filters, hidden_channels)

        else:  # base
            self.lin_geom = nn.Linear(
                num_filters + 2 * hidden_channels, hidden_channels
            )
            self.lin_h = nn.Linear(hidden_channels, hidden_channels)

        if self.complex_mp:
            self.other_mlp = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        if self.mp_type != "simple":
            nn.init.xavier_uniform_(self.lin_geom.weight)
            self.lin_geom.bias.data.fill_(0)
        if self.complex_mp:
            nn.init.xavier_uniform_(self.other_mlp.weight)
            self.other_mlp.bias.data.fill_(0)
        if self.mp_type in {"updownscale", "updownscale_base", "updown_local_env"}:
            nn.init.xavier_uniform_(self.lin_up.weight)
            self.lin_up.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.lin_down.weight)
            self.lin_down.bias.data.fill_(0)
        else:
            nn.init.xavier_uniform_(self.lin_h.weight)
            self.lin_h.bias.data.fill_(0)

    def forward(self, h, edge_index, e):
        """Forward pass of the Interaction block.
        Called in FAENet forward pass to update atom representations.

        Args:
            h (tensor): atom embedddings. (num_atoms, hidden_channels)
            edge_index (tensor): adjacency matrix. (2, num_edges)
            e (tensor): edge embeddings. (num_edges, num_filters)

        Returns:
            (tensor): updated atom embeddings
        """
        # Define edge embedding
        if self.mp_type in {"base", "updownscale_base"}:
            e = torch.cat([e, h[edge_index[0]], h[edge_index[1]]], dim=1)

        if self.mp_type in {
            "updownscale",
            "base",
            "updownscale_base",
        }:
            e = self.act(self.lin_geom(e))

        # --- Message Passing block --

        if self.mp_type == "updownscale" or self.mp_type == "updownscale_base":
            h = self.act(self.lin_down(h))  # downscale node rep.
            h = self.propagate(edge_index, x=h, W=e)  # propagate
            if self.graph_norm:
                h = self.act(self.graph_norm(h))
            h = self.act(self.lin_up(h))  # upscale node rep.

        elif self.mp_type == "updown_local_env":
            h = self.act(self.lin_down(h))
            chi = self.propagate(edge_index, x=h, W=e, local_env=True)
            e = self.lin_geom(e)
            h = self.propagate(edge_index, x=h, W=e)  # propagate
            if self.graph_norm:
                h = self.act(self.graph_norm(h))
            h = torch.cat((h, chi), dim=1)
            h = self.lin_up(h)

        elif self.mp_type in {"base", "simple"}:
            h = self.propagate(edge_index, x=h, W=e)  # propagate
            if self.graph_norm:
                h = self.act(self.graph_norm(h))
            h = self.act(self.lin_h(h))

        else:
            raise ValueError("mp_type provided does not exist")

        if self.complex_mp:
            h = self.act(self.other_mlp(h))

        return h

    def message(self, x_j, W, local_env=None):
        if local_env is not None:
            return W
        else:
            return x_j * W


class OutputBlock(nn.Module):
    """Compute task-specific predictions from final atom representations."""

    def __init__(self, energy_head, hidden_channels, act, out_dim=1):
        super().__init__()
        self.energy_head = energy_head
        self.act = act

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, out_dim)

        if self.energy_head == "weighted-av-final-embeds":
            self.w_lin = Linear(hidden_channels, 1)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        if self.energy_head == "weighted-av-final-embeds":
            nn.init.xavier_uniform_(self.w_lin.weight)
            self.w_lin.bias.data.fill_(0)

    def forward(self, h, edge_index, edge_weight, batch, alpha):
        """Forward pass of the Output block.
        Called in FAENet to make prediction from final atom representations.

        Args:
            h (tensor): atom representations. (num_atoms, hidden_channels)
            edge_index (tensor): adjacency matrix. (2, num_edges)
            edge_weight (tensor): edge weights. (num_edges, )
            batch (tensor): batch indices. (num_atoms, )
            alpha (tensor): atom attention weights for late energy head. (num_atoms, )

        Returns:
            (tensor): graph-level representation (e.g. energy prediction)
        """
        if self.energy_head == "weighted-av-final-embeds":
            alpha = self.w_lin(h)

        # MLP
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if self.energy_head in {
            "weighted-av-initial-embeds",
            "weighted-av-final-embeds",
        }:
            h = h * alpha

        # Global pooling
        out = scatter(h, batch, dim=0, reduce="add")

        return out


class FAENet(BaseModel):
    r"""Non-symmetry preserving GNN model for 3D atomic systems,
    called FAENet: Frame Averaging Equivariant Network.

    Args:
        cutoff (float): Cutoff distance for interatomic interactions.
            (default: :obj:`6.0`)
        preprocess (callable): Pre-processing function for the data. This function
            should accept a data object as input and return a tuple containing the following:
            atomic numbers, batch indices, final adjacency, relative positions, pairwise distances.
            Examples of valid preprocessing functions include `pbc_preprocess`,
            `base_preprocess`, or custom functions.
        act (str): Activation function
            (default: `swish`)
        max_num_neighbors (int): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: `40`)
        hidden_channels (int): Hidden embedding size.
            (default: `128`)
        tag_hidden_channels (int): Hidden tag embedding size.
            (default: :obj:`32`)
        pg_hidden_channels (int): Hidden period and group embedding size.
            (default: :obj:`32`)
        phys_embeds (bool): Do we include fixed physics-aware embeddings.
            (default: :obj: `True`)
        phys_hidden_channels (int): Hidden size of learnable physics-aware embeddings.
            (default: :obj:`0`)
        num_interactions (int): The number of interaction (i.e. message passing) blocks.
            (default: :obj:`4`)
        num_gaussians (int): The number of gaussians :math:`\mu` to encode distance info.
            (default: :obj:`50`)
        num_filters (int): The size of convolutional filters.
            (default: :obj:`128`)
        second_layer_MLP (bool): Use 2-layers MLP at the end of the Embedding block.
            (default: :obj:`False`)
        skip_co (str): Add a skip connection between each interaction block and
            energy-head. (`False`, `"add"`, `"concat"`, `"concat_atom"`)
        mp_type (str): Specificies the Message Passing type of the interaction block.
            (`"base"`, `"updownscale_base"`, `"updownscale"`, `"updown_local_env"`, `"simple"`):
        graph_norm (bool): Whether to apply batch norm after every linear layer.
            (default: :obj:`True`)
        complex_mp (bool); Whether to add a second layer MLP at the end of each Interaction
            (default: :obj:`True`)
        energy_head (str): Method to compute energy prediction
            from atom representations.
            (`None`, `"weighted-av-initial-embeds"`, `"weighted-av-final-embeds"`)
        out_dim (int): size of the output tensor for graph-level predicted properties ("energy")
            Allows to predict multiple properties at the same time.
            (default: :obj:`1`)
        pred_as_dict (bool): Set to False to return a (property) prediction tensor.
            By default, predictions are returned as a dictionary with several keys (e.g. energy, forces)
            (default: :obj:`True`)
        regress_forces (str): Specifies if we predict forces or not, and how
            do we predict them. (`None` or `""`, `"direct"`, `"direct_with_gradient_target"`)
        force_decoder_type (str): Specifies the type of force decoder
            (`"simple"`, `"mlp"`, `"res"`, `"res_updown"`)
        force_decoder_model_config (dict): contains information about the
            for decoder architecture (e.g. number of layers, hidden size).
    """

    def __init__(
        self,
        cutoff: float = 6.0,
        preprocess: Union[str, callable] = "base_preprocess",
        act: str = "swish",
        max_num_neighbors: int = 20,
        hidden_channels: int = 128,
        tag_hidden_channels: int = 0,
        pg_hidden_channels: int = 0,
        phys_embeds: bool = True,
        phys_hidden_channels: int = 0,
        num_interactions: int = 2,
        num_gaussians: int = 20,
        num_filters: int = 64,
        second_layer_MLP: bool = True,
        skip_co: str = "concat",
        mp_type: str = "updownscale_base",
        graph_norm: bool = True,
        complex_mp: bool = False,
        energy_head: Optional[str] = None,
        out_dim: int = 1,
        pred_as_dict: bool = True,
        regress_forces: Optional[str] = None,
        force_decoder_type: Optional[str] = "mlp",
        force_decoder_model_config: Optional[dict] = {"hidden_channels": 64},
    ):
        super(FAENet, self).__init__()

        self.act = act
        self.complex_mp = complex_mp
        self.cutoff = cutoff
        self.energy_head = energy_head
        self.force_decoder_type = force_decoder_type
        self.force_decoder_model_config = force_decoder_model_config
        self.graph_norm = graph_norm
        self.hidden_channels = hidden_channels
        self.max_num_neighbors = max_num_neighbors
        self.mp_type = mp_type
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians
        self.num_interactions = num_interactions
        self.pg_hidden_channels = pg_hidden_channels
        self.phys_embeds = phys_embeds
        self.phys_hidden_channels = phys_hidden_channels
        self.regress_forces = regress_forces
        self.second_layer_MLP = second_layer_MLP
        self.skip_co = skip_co
        self.tag_hidden_channels = tag_hidden_channels
        self.preprocess = preprocess
        self.pred_as_dict = pred_as_dict

        if isinstance(self.preprocess, str):
            self.preprocess = eval(self.preprocess)

        if not isinstance(self.regress_forces, str):
            assert self.regress_forces is False or self.regress_forces is None, (
                "regress_forces must be a string "
                + "('', 'direct', 'direct_with_gradient_target') or False or None"
            )
            self.regress_forces = ""

        if self.mp_type == "simple":
            self.num_filters = self.hidden_channels

        self.act = (
            (getattr(nn.functional, self.act) if self.act != "swish" else swish)
            if isinstance(self.act, str)
            else self.act
        )
        assert callable(self.act), (
            "act must be a callable function or a string "
            + "describing that function in torch.nn.functional"
        )

        # Gaussian Basis
        self.distance_expansion = GaussianSmearing(0.0, self.cutoff, self.num_gaussians)

        # Embedding block
        self.embed_block = EmbeddingBlock(
            self.num_gaussians,
            self.num_filters,
            self.hidden_channels,
            self.tag_hidden_channels,
            self.pg_hidden_channels,
            self.phys_hidden_channels,
            self.phys_embeds,
            self.act,
            self.second_layer_MLP,
        )

        # Interaction block
        self.interaction_blocks = nn.ModuleList(
            [
                InteractionBlock(
                    self.hidden_channels,
                    self.num_filters,
                    self.act,
                    self.mp_type,
                    self.complex_mp,
                    self.graph_norm,
                )
                for _ in range(self.num_interactions)
            ]
        )

        # Output block
        self.output_block = OutputBlock(
            self.energy_head, self.hidden_channels, self.act, out_dim
        )

        # Energy head
        if self.energy_head == "weighted-av-initial-embeds":
            self.w_lin = Linear(self.hidden_channels, 1)

        # Force head
        self.decoder = (
            ForceDecoder(
                self.force_decoder_type,
                self.hidden_channels,
                self.force_decoder_model_config,
                self.act,
            )
            if "direct" in self.regress_forces
            else None
        )

        # Skip co
        if self.skip_co == "concat":
            self.mlp_skip_co = Linear(out_dim * (self.num_interactions + 1), out_dim)
        elif self.skip_co == "concat_atom":
            self.mlp_skip_co = Linear(
                ((self.num_interactions + 1) * self.hidden_channels),
                self.hidden_channels,
            )

    # FAENet's forward pass in done in BaseModel, inherited here.
    # It uses forces_forward() and energy_forward() defined below.

    def forces_forward(self, preds):
        """Predicts forces for 3D atomic systems.
        Can be utilised to predict any atom-level property.

        Args:
            preds (dict): dictionnary with final atomic representations
                (hidden_state) and predicted properties (e.g. energy)
                for each graph

        Returns:
            (dict): additional predicted properties, at an atom-level (e.g. forces)
        """
        if self.decoder:
            return self.decoder(preds["hidden_state"])

    def energy_forward(self, data, preproc=True):
        """Predicts any graph-level property (e.g. energy) for 3D atomic systems.

        Args:
            data (data.Batch): Batch of graphs data objects.
            preproc (bool): Whether to apply (any given) preprocessing to the graph.
                Default to True.

        Returns:
            (dict): predicted properties for each graph (key: "energy")
                and final atomic representations (key: "hidden_state")
        """
        # Pre-process data (e.g. pbc, cutoff graph, etc.)
        # Should output all necessary attributes, in correct format.
        if preproc:
            z, batch, edge_index, rel_pos, edge_weight = self.preprocess(
                data, self.cutoff, self.max_num_neighbors
            )
        else:
            rel_pos = data.pos[data.edge_index[0]] - data.pos[data.edge_index[1]]
            z, batch, edge_index, rel_pos, edge_weight = (
                data.atomic_numbers.long(),
                data.batch,
                data.edge_index,
                rel_pos,
                rel_pos.norm(dim=-1),
            )

        edge_attr = self.distance_expansion(edge_weight)  # RBF of pairwise distances
        assert z.dim() == 1 and z.dtype == torch.long

        # Embedding block
        h, e = self.embed_block(
            z, rel_pos, edge_attr, data.tags if hasattr(data, "tags") else None
        )

        # Compute atom weights for late energy head
        if self.energy_head == "weighted-av-initial-embeds":
            alpha = self.w_lin(h)
        else:
            alpha = None

        # Interaction blocks
        energy_skip_co = []
        for interaction in self.interaction_blocks:
            if self.skip_co == "concat_atom":
                energy_skip_co.append(h)
            elif self.skip_co:
                energy_skip_co.append(
                    self.output_block(h, edge_index, edge_weight, batch, alpha)
                )
            h = h + interaction(h, edge_index, e)

        # Atom skip-co
        if self.skip_co == "concat_atom":
            energy_skip_co.append(h)
            h = self.act(self.mlp_skip_co(torch.cat(energy_skip_co, dim=1)))

        energy = self.output_block(h, edge_index, edge_weight, batch, alpha)

        # Skip-connection
        energy_skip_co.append(energy)
        if self.skip_co == "concat":
            energy = self.mlp_skip_co(torch.cat(energy_skip_co, dim=1))
        elif self.skip_co == "add":
            energy = sum(energy_skip_co)

        preds = {"energy": energy, "hidden_state": h}

        return preds
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

class FAENetMulti(BaseModel):
    r"""Non-symmetry preserving GNN model for 3D atomic systems,
    called FAENet: Frame Averaging Equivariant Network.

    Args:
        cutoff (float): Cutoff distance for interatomic interactions.
            (default: :obj:`6.0`)
        preprocess (callable): Pre-processing function for the data.
        act (str): Activation function (default: `swish`)
        max_num_neighbors (int): Maximum number of neighbors (default: `40`)
        hidden_channels (int): Hidden embedding size (default: `128`)
        tag_hidden_channels (int): Hidden tag embedding size (default: `32`)
        pg_hidden_channels (int): Hidden period and group embedding size (default: `32`)
        phys_embeds (bool): Include fixed physics-aware embeddings (default: `True`)
        phys_hidden_channels (int): Hidden size of learnable physics-aware embeddings (default: `0`)
        num_interactions (int): Number of interaction blocks (default: `4`)
        num_gaussians (int): Number of gaussians to encode distance info (default: `50`)
        num_filters (int): Size of convolutional filters (default: `128`)
        second_layer_MLP (bool): Use 2-layer MLP at the end of the Embedding block (default: `False`)
        skip_co (str): Skip connection type (default: `"concat"`)
        mp_type (str): Message passing type (default: `"updownscale_base"`)
        graph_norm (bool): Apply batch norm after every linear layer (default: `True`)
        complex_mp (bool): Whether to add a second layer MLP at the end of each Interaction (default: `False`)
        energy_head (str): Method to compute energy prediction from atom representations.
        out_dim (int): Size of the output tensor for graph-level predicted properties (default: `1`)
        pred_as_dict (bool): Return predictions as dictionary (default: `True`)
        regress_forces (str): How to predict forces (default: `None`)
        force_decoder_type (str): Type of force decoder (default: `"mlp"`)
        force_decoder_model_config (dict): Config for force decoder architecture.
        clip_encoder (int): Final dimension of processed CLIP embedding (default: `32`)
        clip_input_dim (int): Input dimension for CLIP embedding (default: `768`)
        global_feature_dim (int): Graph-level feature dimension used for fusion (default: `128`)
    """
    def __init__(
        self,
        cutoff: float = 6.0,
        preprocess: Union[str, callable] = "base_preprocess",
        act: str = "swish",
        max_num_neighbors: int = 20,
        hidden_channels: int = 128,
        tag_hidden_channels: int = 0,
        pg_hidden_channels: int = 0,
        phys_embeds: bool = True,
        phys_hidden_channels: int = 0,
        num_interactions: int = 2,
        num_gaussians: int = 20,
        num_filters: int = 64,
        second_layer_MLP: bool = True,
        skip_co: str = "concat",
        mp_type: str = "updownscale_base",
        graph_norm: bool = True,
        complex_mp: bool = False,
        energy_head: Optional[str] = None,
        out_dim: int = 1,
        pred_as_dict: bool = True,
        regress_forces: Optional[str] = None,
        force_decoder_type: Optional[str] = "mlp",
        force_decoder_model_config: Optional[dict] = {"hidden_channels": 64},
        clip_encoder: int = 32,
        clip_input_dim: int = 768,
        global_feature_dim: int = 128
    ):
        super(FAENetMulti, self).__init__()
        self.act = act
        self.complex_mp = complex_mp
        self.cutoff = cutoff
        self.energy_head = energy_head
        self.force_decoder_type = force_decoder_type
        self.force_decoder_model_config = force_decoder_model_config
        self.graph_norm = graph_norm
        self.hidden_channels = hidden_channels
        self.max_num_neighbors = max_num_neighbors
        self.mp_type = mp_type
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians
        self.num_interactions = num_interactions
        self.pg_hidden_channels = pg_hidden_channels
        self.phys_embeds = phys_embeds
        self.phys_hidden_channels = phys_hidden_channels
        self.regress_forces = regress_forces
        self.second_layer_MLP = second_layer_MLP
        self.skip_co = skip_co
        self.tag_hidden_channels = tag_hidden_channels
        self.preprocess = preprocess
        self.pred_as_dict = pred_as_dict

        if isinstance(self.preprocess, str):
            self.preprocess = eval(self.preprocess)

        if not isinstance(self.regress_forces, str):
            assert self.regress_forces is False or self.regress_forces is None, (
                "regress_forces must be a string or False/None"
            )
            self.regress_forces = ""

        if self.mp_type == "simple":
            self.num_filters = self.hidden_channels

        self.act = (
            (getattr(nn.functional, self.act) if self.act != "swish" else swish)
            if isinstance(self.act, str)
            else self.act
        )
        assert callable(self.act), "act must be callable"

        # Gaussian Basis
        self.distance_expansion = GaussianSmearing(0.0, self.cutoff, self.num_gaussians)

        # Embedding block
        self.embed_block = EmbeddingBlock(
            self.num_gaussians,
            self.num_filters,
            self.hidden_channels,
            self.tag_hidden_channels,
            self.pg_hidden_channels,
            self.phys_hidden_channels,
            self.phys_embeds,
            self.act,
            self.second_layer_MLP,
        )

        # Interaction blocks
        self.interaction_blocks = nn.ModuleList(
            [
                InteractionBlock(
                    self.hidden_channels,
                    self.num_filters,
                    self.act,
                    self.mp_type,
                    self.complex_mp,
                    self.graph_norm,
                )
                for _ in range(self.num_interactions)
            ]
        )

        # Output block
        self.output_block = OutputBlock(
            self.energy_head, self.hidden_channels, self.act, out_dim
        )

        # Energy head
        if self.energy_head == "weighted-av-initial-embeds":
            self.w_lin = nn.Linear(self.hidden_channels, 1)

        # Force head
        self.decoder = (
            ForceDecoder(
                self.force_decoder_type,
                self.hidden_channels,
                self.force_decoder_model_config,
                self.act,
            )
            if "direct" in self.regress_forces
            else None
        )

        # Skip connection
        if self.skip_co == "concat":
            self.mlp_skip_co = nn.Linear(out_dim * (self.num_interactions + 1), out_dim)
        elif self.skip_co == "concat_atom":
            self.mlp_skip_co = nn.Linear(
                ((self.num_interactions + 1) * self.hidden_channels),
                self.hidden_channels,
            )

        # --- New: Clip branch and gated fusion for FAENet ---
        self.clip_encoder = clip_encoder
        self.clip_input_dim = clip_input_dim
        self.global_feature_dim = global_feature_dim
        # Define a clip head that processes clip embeddings from clip_input_dim to an intermediate size (e.g., 256)
        # then maps to clip_encoder.
        self.clip_head = nn.Sequential(
            nn.Linear(self.clip_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.clip_encoder)
        )
        # We assume that a global (graph-level) representation is obtained by global pooling
        # of the atom-level features (which we will project to global_feature_dim if needed).
        # Define the gated fusion module:
        self.gated_fusion = GatedFusion(
            mol_dim=self.global_feature_dim, clip_dim=self.clip_encoder, fusion_dim=self.global_feature_dim
        )
        # Define a final energy head that maps the fused global features to a scalar.
        self.final_energy_head = nn.Linear(self.global_feature_dim, 1)
        # --- End new section ---

    def forward(self, data, clip_embeddings=None):
        return self.energy_forward(data, preproc=True, clip_embeddings=clip_embeddings)

    def forces_forward(self, preds):
        if self.decoder:
            return self.decoder(preds["hidden_state"])

    def energy_forward(self, data, preproc=True, clip_embeddings=None):
        """Predicts energy for 3D atomic systems with additional gated fusion of CLIP embeddings.

        Args:
            data (data.Batch): Batch of graph data objects.
            preproc (bool): Whether to apply the preprocessing function.
            clip_embeddings (Tensor, optional): CLIP embeddings for each graph [B, clip_input_dim].

        Returns:
            dict: Contains predicted energy and final atomic representations.
        """
        # Pre-process data.
        if preproc:
            z, batch, edge_index, rel_pos, edge_weight = self.preprocess(
                data, self.cutoff, self.max_num_neighbors
            )
        else:
            rel_pos = data.pos[data.edge_index[0]] - data.pos[data.edge_index[1]]
            z, batch, edge_index, rel_pos, edge_weight = (
                data.atomic_numbers.long(),
                data.batch,
                data.edge_index,
                rel_pos,
                rel_pos.norm(dim=-1),
            )

        edge_attr = self.distance_expansion(edge_weight)  # RBF encoding
        assert z.dim() == 1 and z.dtype == torch.long

        # Embedding block.
        h, e = self.embed_block(
            z, rel_pos, edge_attr, data.tags if hasattr(data, "tags") else None
        )

        # Compute alpha if using weighted-average energy head.
        if self.energy_head == "weighted-av-initial-embeds":
            alpha = self.w_lin(h)
        else:
            alpha = None

        # Original skip-connection based energy computation.
        energy_skip_co = []
        for interaction in self.interaction_blocks:
            if self.skip_co == "concat_atom":
                energy_skip_co.append(h)
            elif self.skip_co:
                energy_skip_co.append(
                    self.output_block(h, edge_index, edge_weight, batch, alpha)
                )
            h = h + interaction(h, edge_index, e)
        if self.skip_co == "concat_atom":
            energy_skip_co.append(h)
            h = self.act(self.mlp_skip_co(torch.cat(energy_skip_co, dim=1)))
        energy_orig = self.output_block(h, edge_index, edge_weight, batch, alpha)
        energy_skip_co.append(energy_orig)
        if self.skip_co == "concat":
            energy_orig = self.mlp_skip_co(torch.cat(energy_skip_co, dim=1))
        elif self.skip_co == "add":
            energy_orig = sum(energy_skip_co)

        # --- New: Gated fusion with CLIP embeddings at graph level ---
        # Obtain a graph-level representation from atom-level features.
        global_feature = global_add_pool(h, batch)  # [B, hidden_channels]
        # Project to desired global_feature_dim if needed.
        if global_feature.size(1) != self.global_feature_dim:
            global_feature = nn.functional.pad(
                global_feature,
                (0, self.global_feature_dim - global_feature.size(1)),
                mode='constant',
                value=0
            )
        if clip_embeddings is not None:
            B = global_feature.size(0)
            clip_embeddings = clip_embeddings.view(B, self.clip_input_dim)
            clip_feat = self.clip_head(clip_embeddings)  # [B, clip_encoder]
            global_feature = self.gated_fusion(global_feature, clip_feat)
        energy_clip = self.final_energy_head(global_feature)
        # --- End new section ---

        # Combine the original energy prediction and the CLIP-fused prediction (e.g., by averaging).
        energy = (energy_orig + energy_clip) / 2.0

        preds = {"energy": energy, "hidden_state": h}
        return preds