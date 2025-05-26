import numpy as np
import os
import os.path as osp
import torch
from functools import partial
from math import pi as PI
from math import sqrt
from torch import Tensor
from torch.nn import Embedding
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.data import Dataset, download_url
from torch_geometric.nn import radius_graph
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor, SparseTensor
from torch_geometric.utils import scatter
from typing import Callable, Dict, Optional, Tuple, Union
import torch.nn as nn
qm9_target_dict: Dict[int, str] = {
    0: 'mu',
    1: 'alpha',
    2: 'homo',
    3: 'lumo',
    5: 'r2',
    6: 'zpve',
    7: 'U0',
    8: 'U',
    9: 'H',
    10: 'G',
    11: 'Cv',
}


class Envelope(torch.nn.Module):
    def __init__(self, exponent: int):
        super().__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x: Tensor) -> Tensor:
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return (1.0 / x + a * x_pow_p0 + b * x_pow_p1 +
                c * x_pow_p2) * (x < 1.0).to(x.dtype)


class BesselBasisLayer(torch.nn.Module):
    def __init__(self, num_radial: int, cutoff: float = 5.0,
                 envelope_exponent: int = 5):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = torch.nn.Parameter(torch.empty(num_radial))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)
        self.freq.requires_grad_()

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()


class SphericalBasisLayer(torch.nn.Module):
    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        envelope_exponent: int = 5,
    ):
        super().__init__()
        import sympy as sym

        from torch_geometric.nn.models.dimenet_utils import (
            bessel_basis,
            real_sph_harm,
        )

        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(partial(self._sph_to_tensor, sph1))
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    @staticmethod
    def _sph_to_tensor(sph, x: Tensor) -> Tensor:
        return torch.zeros_like(x) + sph

    def forward(self, dist: Tensor, angle: Tensor, idx_kj: Tensor) -> Tensor:
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, num_radial: int, hidden_channels: int, act: Callable):
        super().__init__()
        self.act = act

        self.emb = Embedding(95, hidden_channels)
        self.lin_rbf = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x: Tensor, rbf: Tensor, i: Tensor, j: Tensor) -> Tensor:
        x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        return self.act(self.lin(torch.cat([x[i], x[j], rbf], dim=-1)))


class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels: int, act: Callable):
        super().__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class InteractionBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_bilinear: int,
        num_spherical: int,
        num_radial: int,
        num_before_skip: int,
        num_after_skip: int,
        act: Callable,
    ):
        super().__init__()
        self.act = act

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)
        self.lin_sbf = Linear(num_spherical * num_radial, num_bilinear,
                              bias=False)

        # Dense transformations of input messages.
        self.lin_kj = Linear(hidden_channels, hidden_channels)
        self.lin_ji = Linear(hidden_channels, hidden_channels)

        self.W = torch.nn.Parameter(
            torch.empty(hidden_channels, num_bilinear, hidden_channels))

        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
        ])
        self.lin = Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)
        self.W.data.normal_(mean=0, std=2 / self.W.size(0))
        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

    def forward(self, x: Tensor, rbf: Tensor, sbf: Tensor, idx_kj: Tensor,
                idx_ji: Tensor) -> Tensor:
        rbf = self.lin_rbf(rbf)
        sbf = self.lin_sbf(sbf)

        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))
        x_kj = x_kj * rbf
        x_kj = torch.einsum('wj,wl,ijl->wi', sbf, x_kj[idx_kj], self.W)
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0), reduce='sum')

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h


class InteractionPPBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        int_emb_size: int,
        basis_emb_size: int,
        num_spherical: int,
        num_radial: int,
        num_before_skip: int,
        num_after_skip: int,
        act: Callable,
    ):
        super().__init__()
        self.act = act

        # Transformation of Bessel and spherical basis representations:
        self.lin_rbf1 = Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = Linear(basis_emb_size, hidden_channels, bias=False)

        self.lin_sbf1 = Linear(num_spherical * num_radial, basis_emb_size,
                               bias=False)
        self.lin_sbf2 = Linear(basis_emb_size, int_emb_size, bias=False)

        # Hidden transformation of input message:
        self.lin_kj = Linear(hidden_channels, hidden_channels)
        self.lin_ji = Linear(hidden_channels, hidden_channels)

        # Embedding projections for interaction triplets:
        self.lin_down = Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = Linear(int_emb_size, hidden_channels, bias=False)

        # Residual layers before and after skip connection:
        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
        ])
        self.lin = Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

    def forward(self, x: Tensor, rbf: Tensor, sbf: Tensor, idx_kj: Tensor,
                idx_ji: Tensor) -> Tensor:
        # Initial transformation:
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))

        # Transformation via Bessel basis:
        rbf = self.lin_rbf1(rbf)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        # Down project embedding and generating triple-interactions:
        x_kj = self.act(self.lin_down(x_kj))

        # Transform via 2D spherical basis:
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        # Aggregate interactions and up-project embeddings:
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0), reduce='sum')
        x_kj = self.act(self.lin_up(x_kj))

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h


class OutputBlock(torch.nn.Module):
    def __init__(
        self,
        num_radial: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        act: Callable,
        output_initializer: str = 'zeros',
    ):
        assert output_initializer in {'zeros', 'glorot_orthogonal'}

        super().__init__()

        self.act = act
        self.output_initializer = output_initializer

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lin = Linear(hidden_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        if self.output_initializer == 'zeros':
            self.lin.weight.data.fill_(0)
        elif self.output_initializer == 'glorot_orthogonal':
            glorot_orthogonal(self.lin.weight, scale=2.0)

    def forward(self, x: Tensor, rbf: Tensor, i: Tensor,
                num_nodes: Optional[int] = None) -> Tensor:
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes, reduce='sum')
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


class OutputPPBlock(torch.nn.Module):
    def __init__(
        self,
        num_radial: int,
        hidden_channels: int,
        out_emb_channels: int,
        out_channels: int,
        num_layers: int,
        act: Callable,
        output_initializer: str = 'zeros',
    ):
        assert output_initializer in {'zeros', 'glorot_orthogonal'}

        super().__init__()

        self.act = act
        self.output_initializer = output_initializer

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)

        # The up-projection layer:
        self.lin_up = Linear(hidden_channels, out_emb_channels, bias=False)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(out_emb_channels, out_emb_channels))
        self.lin = Linear(out_emb_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        if self.output_initializer == 'zeros':
            self.lin.weight.data.fill_(0)
        elif self.output_initializer == 'glorot_orthogonal':
            glorot_orthogonal(self.lin.weight, scale=2.0)

    def forward(self, x: Tensor, rbf: Tensor, i: Tensor,
                num_nodes: Optional[int] = None) -> Tensor:
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes, reduce='sum')
        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


def triplets(
    edge_index: Tensor,
    num_nodes: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    row, col = edge_index  # j->i

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k  # Remove i == k triplets.
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji


class DimeNet(torch.nn.Module):
    r"""The directional message passing neural network (DimeNet) from the
    `"Directional Message Passing for Molecular Graphs"
    <https://arxiv.org/abs/2003.03123>`_ paper.
    DimeNet transforms messages based on the angle between them in a
    rotation-equivariant fashion.

    .. note::

        For an example of using a pretrained DimeNet variant, see
        `examples/qm9_pretrained_dimenet.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        qm9_pretrained_dimenet.py>`_.

    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        num_bilinear (int): Size of the bilinear layer tensor.
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act (str or Callable, optional): The activation function.
            (default: :obj:`"swish"`)
        output_initializer (str, optional): The initialization method for the
            output layer (:obj:`"zeros"`, :obj:`"glorot_orthogonal"`).
            (default: :obj:`"zeros"`)
    """

    url = ('https://github.com/klicperajo/dimenet/raw/master/pretrained/'
           'dimenet')

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_blocks: int,
        num_bilinear: int,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: Union[str, Callable] = 'swish',
        output_initializer: str = 'zeros',
    ):
        super().__init__()

        if num_spherical < 2:
            raise ValueError("'num_spherical' should be greater than 1")

        act = activation_resolver(act)

        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.num_blocks = num_blocks

        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff,
                                       envelope_exponent)

        self.emb = EmbeddingBlock(num_radial, hidden_channels, act)

        self.output_blocks = torch.nn.ModuleList([
            OutputBlock(
                num_radial,
                hidden_channels,
                out_channels,
                num_output_layers,
                act,
                output_initializer,
            ) for _ in range(num_blocks + 1)
        ])

        self.interaction_blocks = torch.nn.ModuleList([
            InteractionBlock(
                hidden_channels,
                num_bilinear,
                num_spherical,
                num_radial,
                num_before_skip,
                num_after_skip,
                act,
            ) for _ in range(num_blocks)
        ])

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.rbf.reset_parameters()
        self.emb.reset_parameters()
        for out in self.output_blocks:
            out.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()

    @classmethod
    def from_qm9_pretrained(
        cls,
        root: str,
        dataset: Dataset,
        target: int,
    ) -> Tuple['DimeNet', Dataset, Dataset, Dataset]:  # pragma: no cover
        r"""Returns a pre-trained :class:`DimeNet` model on the
        :class:`~torch_geometric.datasets.QM9` dataset, trained on the
        specified target :obj:`target`.
        """
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf

        assert target >= 0 and target <= 12 and not target == 4

        root = osp.expanduser(osp.normpath(root))
        path = osp.join(root, 'pretrained_dimenet', qm9_target_dict[target])

        os.makedirs(path, exist_ok=True)
        url = f'{cls.url}/{qm9_target_dict[target]}'

        if not osp.exists(osp.join(path, 'checkpoint')):
            download_url(f'{url}/checkpoint', path)
            download_url(f'{url}/ckpt.data-00000-of-00002', path)
            download_url(f'{url}/ckpt.data-00001-of-00002', path)
            download_url(f'{url}/ckpt.index', path)

        path = osp.join(path, 'ckpt')
        reader = tf.train.load_checkpoint(path)

        model = cls(
            hidden_channels=128,
            out_channels=1,
            num_blocks=6,
            num_bilinear=8,
            num_spherical=7,
            num_radial=6,
            cutoff=5.0,
            envelope_exponent=5,
            num_before_skip=1,
            num_after_skip=2,
            num_output_layers=3,
        )

        def copy_(src, name, transpose=False):
            init = reader.get_tensor(f'{name}/.ATTRIBUTES/VARIABLE_VALUE')
            init = torch.from_numpy(init)
            if name[-6:] == 'kernel':
                init = init.t()
            src.data.copy_(init)

        copy_(model.rbf.freq, 'rbf_layer/frequencies')
        copy_(model.emb.emb.weight, 'emb_block/embeddings')
        copy_(model.emb.lin_rbf.weight, 'emb_block/dense_rbf/kernel')
        copy_(model.emb.lin_rbf.bias, 'emb_block/dense_rbf/bias')
        copy_(model.emb.lin.weight, 'emb_block/dense/kernel')
        copy_(model.emb.lin.bias, 'emb_block/dense/bias')

        for i, block in enumerate(model.output_blocks):
            copy_(block.lin_rbf.weight, f'output_blocks/{i}/dense_rbf/kernel')
            for j, lin in enumerate(block.lins):
                copy_(lin.weight, f'output_blocks/{i}/dense_layers/{j}/kernel')
                copy_(lin.bias, f'output_blocks/{i}/dense_layers/{j}/bias')
            copy_(block.lin.weight, f'output_blocks/{i}/dense_final/kernel')

        for i, block in enumerate(model.interaction_blocks):
            copy_(block.lin_rbf.weight, f'int_blocks/{i}/dense_rbf/kernel')
            copy_(block.lin_sbf.weight, f'int_blocks/{i}/dense_sbf/kernel')
            copy_(block.lin_kj.weight, f'int_blocks/{i}/dense_kj/kernel')
            copy_(block.lin_kj.bias, f'int_blocks/{i}/dense_kj/bias')
            copy_(block.lin_ji.weight, f'int_blocks/{i}/dense_ji/kernel')
            copy_(block.lin_ji.bias, f'int_blocks/{i}/dense_ji/bias')
            copy_(block.W, f'int_blocks/{i}/bilinear')
            for j, layer in enumerate(block.layers_before_skip):
                copy_(layer.lin1.weight,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_1/kernel')
                copy_(layer.lin1.bias,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_1/bias')
                copy_(layer.lin2.weight,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_2/kernel')
                copy_(layer.lin2.bias,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_2/bias')
            copy_(block.lin.weight, f'int_blocks/{i}/final_before_skip/kernel')
            copy_(block.lin.bias, f'int_blocks/{i}/final_before_skip/bias')
            for j, layer in enumerate(block.layers_after_skip):
                copy_(layer.lin1.weight,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_1/kernel')
                copy_(layer.lin1.bias,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_1/bias')
                copy_(layer.lin2.weight,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_2/kernel')
                copy_(layer.lin2.bias,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_2/bias')

        # Use the same random seed as the official DimeNet` implementation.
        random_state = np.random.RandomState(seed=42)
        perm = torch.from_numpy(random_state.permutation(np.arange(130831)))
        perm = perm.long()
        train_idx = perm[:110000]
        val_idx = perm[110000:120000]
        test_idx = perm[120000:]

        return model, (dataset[train_idx], dataset[val_idx], dataset[test_idx])

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: OptTensor = None,
    ) -> Tensor:
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
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            edge_index, num_nodes=z.size(0))

        # Calculate distances.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        if isinstance(self, DimeNetPlusPlus):
            pos_jk, pos_ij = pos[idx_j] - pos[idx_k], pos[idx_i] - pos[idx_j]
            a = (pos_ij * pos_jk).sum(dim=-1)
            b = torch.cross(pos_ij, pos_jk, dim=1).norm(dim=-1)
        elif isinstance(self, DimeNet):
            pos_ji, pos_ki = pos[idx_j] - pos[idx_i], pos[idx_k] - pos[idx_i]
            a = (pos_ji * pos_ki).sum(dim=-1)
            b = torch.cross(pos_ji, pos_ki, dim=1).norm(dim=-1)
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
            return scatter(P, batch, dim=0, reduce='sum')


class DimeNetPlusPlus(DimeNet):
    r"""The DimeNet++ from the `"Fast and Uncertainty-Aware
    Directional Message Passing for Non-Equilibrium Molecules"
    <https://arxiv.org/abs/2011.14115>`_ paper.

    :class:`DimeNetPlusPlus` is an upgrade to the :class:`DimeNet` model with
    8x faster and 10% more accurate than :class:`DimeNet`.

    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        int_emb_size (int): Size of embedding in the interaction block.
        basis_emb_size (int): Size of basis embedding in the interaction block.
        out_emb_channels (int): Size of embedding in the output block.
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff: (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip: (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip: (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers: (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act: (str or Callable, optional): The activation funtion.
            (default: :obj:`"swish"`)
        output_initializer (str, optional): The initialization method for the
            output layer (:obj:`"zeros"`, :obj:`"glorot_orthogonal"`).
            (default: :obj:`"zeros"`)
    """

    url = ('https://raw.githubusercontent.com/gasteigerjo/dimenet/'
           'master/pretrained/dimenet_pp')

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_blocks: int,
        int_emb_size: int,
        basis_emb_size: int,
        out_emb_channels: int,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: Union[str, Callable] = 'swish',
        output_initializer: str = 'zeros',
    ):
        act = activation_resolver(act)

        super().__init__(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            num_bilinear=1,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
            act=act,
            output_initializer=output_initializer,
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
                output_initializer,
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

        self.reset_parameters()

    @classmethod
    def from_qm9_pretrained(
        cls,
        root: str,
        dataset: Dataset,
        target: int,
    ) -> Tuple['DimeNetPlusPlus', Dataset, Dataset,
               Dataset]:  # pragma: no cover
        r"""Returns a pre-trained :class:`DimeNetPlusPlus` model on the
        :class:`~torch_geometric.datasets.QM9` dataset, trained on the
        specified target :obj:`target`.
        """
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf

        assert target >= 0 and target <= 12 and not target == 4

        root = osp.expanduser(osp.normpath(root))
        path = osp.join(root, 'pretrained_dimenet_pp', qm9_target_dict[target])

        os.makedirs(path, exist_ok=True)
        url = f'{cls.url}/{qm9_target_dict[target]}'

        if not osp.exists(osp.join(path, 'checkpoint')):
            download_url(f'{url}/checkpoint', path)
            download_url(f'{url}/ckpt.data-00000-of-00002', path)
            download_url(f'{url}/ckpt.data-00001-of-00002', path)
            download_url(f'{url}/ckpt.index', path)

        path = osp.join(path, 'ckpt')
        reader = tf.train.load_checkpoint(path)

        # Configuration from DimeNet++:
        # https://github.com/gasteigerjo/dimenet/blob/master/config_pp.yaml
        model = cls(
            hidden_channels=128,
            out_channels=1,
            num_blocks=4,
            int_emb_size=64,
            basis_emb_size=8,
            out_emb_channels=256,
            num_spherical=7,
            num_radial=6,
            cutoff=5.0,
            max_num_neighbors=32,
            envelope_exponent=5,
            num_before_skip=1,
            num_after_skip=2,
            num_output_layers=3,
        )

        def copy_(src, name, transpose=False):
            init = reader.get_tensor(f'{name}/.ATTRIBUTES/VARIABLE_VALUE')
            init = torch.from_numpy(init)
            if name[-6:] == 'kernel':
                init = init.t()
            src.data.copy_(init)

        copy_(model.rbf.freq, 'rbf_layer/frequencies')
        copy_(model.emb.emb.weight, 'emb_block/embeddings')
        copy_(model.emb.lin_rbf.weight, 'emb_block/dense_rbf/kernel')
        copy_(model.emb.lin_rbf.bias, 'emb_block/dense_rbf/bias')
        copy_(model.emb.lin.weight, 'emb_block/dense/kernel')
        copy_(model.emb.lin.bias, 'emb_block/dense/bias')

        for i, block in enumerate(model.output_blocks):
            copy_(block.lin_rbf.weight, f'output_blocks/{i}/dense_rbf/kernel')
            copy_(block.lin_up.weight,
                  f'output_blocks/{i}/up_projection/kernel')
            for j, lin in enumerate(block.lins):
                copy_(lin.weight, f'output_blocks/{i}/dense_layers/{j}/kernel')
                copy_(lin.bias, f'output_blocks/{i}/dense_layers/{j}/bias')
            copy_(block.lin.weight, f'output_blocks/{i}/dense_final/kernel')

        for i, block in enumerate(model.interaction_blocks):
            copy_(block.lin_rbf1.weight, f'int_blocks/{i}/dense_rbf1/kernel')
            copy_(block.lin_rbf2.weight, f'int_blocks/{i}/dense_rbf2/kernel')
            copy_(block.lin_sbf1.weight, f'int_blocks/{i}/dense_sbf1/kernel')
            copy_(block.lin_sbf2.weight, f'int_blocks/{i}/dense_sbf2/kernel')

            copy_(block.lin_ji.weight, f'int_blocks/{i}/dense_ji/kernel')
            copy_(block.lin_ji.bias, f'int_blocks/{i}/dense_ji/bias')
            copy_(block.lin_kj.weight, f'int_blocks/{i}/dense_kj/kernel')
            copy_(block.lin_kj.bias, f'int_blocks/{i}/dense_kj/bias')

            copy_(block.lin_down.weight,
                  f'int_blocks/{i}/down_projection/kernel')
            copy_(block.lin_up.weight, f'int_blocks/{i}/up_projection/kernel')

            for j, layer in enumerate(block.layers_before_skip):
                copy_(layer.lin1.weight,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_1/kernel')
                copy_(layer.lin1.bias,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_1/bias')
                copy_(layer.lin2.weight,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_2/kernel')
                copy_(layer.lin2.bias,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_2/bias')

            copy_(block.lin.weight, f'int_blocks/{i}/final_before_skip/kernel')
            copy_(block.lin.bias, f'int_blocks/{i}/final_before_skip/bias')

            for j, layer in enumerate(block.layers_after_skip):
                copy_(layer.lin1.weight,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_1/kernel')
                copy_(layer.lin1.bias,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_1/bias')
                copy_(layer.lin2.weight,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_2/kernel')
                copy_(layer.lin2.bias,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_2/bias')

        random_state = np.random.RandomState(seed=42)
        perm = torch.from_numpy(random_state.permutation(np.arange(130831)))
        perm = perm.long()
        train_idx = perm[:110000]
        val_idx = perm[110000:120000]
        test_idx = perm[120000:]

        return model, (dataset[train_idx], dataset[val_idx], dataset[test_idx])

#
# class DimeNetMulti(DimeNetPlusPlus):
#     def __init__(
#             self,
#             hidden_channels: int,
#             num_blocks: int,
#             int_emb_size: int,
#             basis_emb_size: int,
#             out_emb_channels: int,
#             num_spherical: int,
#             num_radial: int,
#             clip_encoder: int = 32,
#             cutoff: float = 5.0,
#             max_num_neighbors: int = 32,
#             envelope_exponent: int = 5,
#             num_before_skip: int = 1,
#             num_after_skip: int = 2,
#             num_output_layers: int = 3,
#             act: Union[str, Callable] = 'swish',
#             output_initializer: str = 'zeros',
#     ):
#         # We override the original out_channels: rather than output a scalar directly,
#         # we want an intermediate molecular representation of dimension hidden_channels.
#         super().__init__(
#             hidden_channels=hidden_channels,
#             out_channels=hidden_channels,  # intermediate representation dimension
#             num_blocks=num_blocks,
#             int_emb_size=int_emb_size,
#             basis_emb_size=basis_emb_size,
#             out_emb_channels=out_emb_channels,
#             num_spherical=num_spherical,
#             num_radial=num_radial,
#             cutoff=cutoff,
#             max_num_neighbors=max_num_neighbors,
#             envelope_exponent=envelope_exponent,
#             num_before_skip=num_before_skip,
#             num_after_skip=num_after_skip,
#             num_output_layers=num_output_layers,
#             act=act,
#             output_initializer=output_initializer,
#         )
#         self.clip_encoder = clip_encoder
#
#         # Branch to process 512-dim CLIP embeddings.
#         self.clip_head = Sequential(
#             Linear(512, hidden_channels),
#             ReLU(),
#             Linear(hidden_channels, clip_encoder)
#         )
#         # Define FiLM parameters: project clip features to scaling and shifting vectors.
#         self.film_gamma = Linear(clip_encoder, hidden_channels)
#         self.film_beta = Linear(clip_encoder, hidden_channels)
#
#         # Final prediction layer.
#         self.lin_target = Linear(hidden_channels, 1)
#
#     def forward(self, z: Tensor, pos: Tensor, batch: Optional[Tensor] = None,
#                 clip_embeddings: Optional[Tensor] = None) -> Tensor:
#         """
#         Forward pass with optional CLIP embeddings.
#
#         Args:
#             z (Tensor): Atomic numbers [num_atoms].
#             pos (Tensor): Atomic positions [num_atoms, 3].
#             batch (Tensor, optional): Batch indices for atoms.
#             clip_embeddings (Tensor, optional): CLIP embeddings for each molecule
#                 [batch_size, 512].
#
#         Returns:
#             Tensor: Final target prediction [batch_size, 1].
#         """
#         # Create interaction graph.
#         edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
#                                   max_num_neighbors=self.max_num_neighbors)
#         i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
#             edge_index, num_nodes=z.size(0))
#         # Compute distances.
#         dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
#         # Compute angles (using the DimeNet++ variant formula).
#         pos_jk, pos_ij = pos[idx_j] - pos[idx_k], pos[idx_i] - pos[idx_j]
#         a = (pos_ij * pos_jk).sum(dim=-1)
#         b = torch.cross(pos_ij, pos_jk, dim=1).norm(dim=-1)
#         angle = torch.atan2(b, a)
#         # Compute radial and spherical basis functions.
#         rbf = self.rbf(dist)
#         sbf = self.sbf(dist, angle, idx_kj)
#         # Embedding block.
#         x = self.emb(z, rbf, i, j)
#         # First output block produces an initial molecular representation.
#         h = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))
#         # Propagate through interaction and output blocks.
#         for interaction_block, output_block in zip(self.interaction_blocks,
#                                                    self.output_blocks[1:]):
#             x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
#             h = h + output_block(x, rbf, i, num_nodes=pos.size(0))
#         # Aggregate node (or atom) features into a molecule-level representation.
#         if batch is not None:
#             h = scatter(h, batch, dim=0, reduce='sum')
#         else:
#             h = h.sum(dim=0, keepdim=True)
#
#         # Process CLIP embeddings using FiLM fusion.
#         if clip_embeddings is not None:
#             B = h.size(0)
#             clip_embeddings = clip_embeddings.view(B, 512)
#             clip_feat = self.clip_head(clip_embeddings)  # shape: [B, clip_encoder]
#             # Generate FiLM parameters.
#             gamma = self.film_gamma(clip_feat)  # scaling factors [B, hidden_channels]
#             beta = self.film_beta(clip_feat)  # shifting factors [B, hidden_channels]
#             # FiLM fusion: modulate the molecular representation.
#             h = gamma * h + beta
#
#         # Final prediction.
#         out = self.lin_target(h)
#         return out

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
        # Multihead attention module (batch_first=True so inputs are [B, seq_len, attn_dim]).
        self.attn = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=num_heads, batch_first=True)
        # Project the attended output back to the molecular dimension.
        self.out_proj = Linear(attn_dim, mol_dim)

    def forward(self, h: torch.Tensor, clip_feat: torch.Tensor) -> torch.Tensor:
        # h: [B, mol_dim]; clip_feat: [B, clip_dim]
        # Project and unsqueeze to add a sequence dimension.
        h_proj = self.mol_proj(h).unsqueeze(1)  # [B, 1, attn_dim]
        clip_proj = self.clip_proj(clip_feat).unsqueeze(1)  # [B, 1, attn_dim]
        # Use clip features as both key and value, and h as the query.
        attn_out, _ = self.attn(query=h_proj, key=clip_proj, value=clip_proj)
        attn_out = attn_out.squeeze(1)  # [B, attn_dim]
        fusion = self.out_proj(attn_out)  # [B, mol_dim]
        # Residual connection.
        return h + fusion

#
# # ---------------------
# # Updated DimeNetMulti with Cross-Modal Attention Fusion
# # ---------------------
# class DimeNetMulti(DimeNetPlusPlus):
#     def __init__(
#             self,
#             hidden_channels: int,
#             num_blocks: int,
#             int_emb_size: int,
#             basis_emb_size: int,
#             out_emb_channels: int,
#             num_spherical: int,
#             num_radial: int,
#             clip_encoder: int = 32,
#             cutoff: float = 5.0,
#             max_num_neighbors: int = 32,
#             envelope_exponent: int = 5,
#             num_before_skip: int = 1,
#             num_after_skip: int = 2,
#             num_output_layers: int = 3,
#             act: Union[str, Callable] = 'swish',
#             output_initializer: str = 'zeros',
#     ):
#         # We override the original out_channels: rather than output a scalar directly,
#         # we want an intermediate molecular representation of dimension hidden_channels.
#         super().__init__(
#             hidden_channels=hidden_channels,
#             out_channels=hidden_channels,  # intermediate representation dimension
#             num_blocks=num_blocks,
#             int_emb_size=int_emb_size,
#             basis_emb_size=basis_emb_size,
#             out_emb_channels=out_emb_channels,
#             num_spherical=num_spherical,
#             num_radial=num_radial,
#             cutoff=cutoff,
#             max_num_neighbors=max_num_neighbors,
#             envelope_exponent=envelope_exponent,
#             num_before_skip=num_before_skip,
#             num_after_skip=num_after_skip,
#             num_output_layers=num_output_layers,
#             act=act,
#             output_initializer=output_initializer,
#         )
#         self.clip_encoder = clip_encoder
#
#         # Branch to process 512-dim CLIP embeddings.
#         self.clip_head = Sequential(
#             Linear(768, hidden_channels),
#             ReLU(),
#             Linear(hidden_channels, clip_encoder)
#         )
#         # Replace FiLM fusion with cross-attention fusion.
#         self.cross_attn_fusion = CrossAttentionFusion(
#             mol_dim=hidden_channels,
#             clip_dim=clip_encoder,
#             attn_dim=128,
#             num_heads=4
#         )
#         # Final prediction layer.
#         self.lin_target = Linear(hidden_channels, 1)
#
#     def forward(self, z: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None,
#                 clip_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
#         """
#         Forward pass with optional CLIP embeddings.
#
#         Args:
#             z (Tensor): Atomic numbers [num_atoms].
#             pos (Tensor): Atomic positions [num_atoms, 3].
#             batch (Tensor, optional): Batch indices for atoms.
#             clip_embeddings (Tensor, optional): CLIP embeddings for each molecule [batch_size, 512].
#
#         Returns:
#             Tensor: Final target prediction [batch_size, 1].
#         """
#         # Create interaction graph.
#         edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
#                                   max_num_neighbors=self.max_num_neighbors)
#         i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
#             edge_index, num_nodes=z.size(0))
#         # Compute distances.
#         dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
#         # Compute angles (using the DimeNet++ variant formula).
#         pos_jk, pos_ij = pos[idx_j] - pos[idx_k], pos[idx_i] - pos[idx_j]
#         a = (pos_ij * pos_jk).sum(dim=-1)
#         b = torch.cross(pos_ij, pos_jk, dim=1).norm(dim=-1)
#         angle = torch.atan2(b, a)
#         # Compute radial and spherical basis functions.
#         rbf = self.rbf(dist)
#         sbf = self.sbf(dist, angle, idx_kj)
#         # Embedding block.
#         x = self.emb(z, rbf, i, j)
#         # First output block produces an initial molecular representation.
#         h = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))
#         # Propagate through interaction and output blocks.
#         for interaction_block, output_block in zip(self.interaction_blocks,
#                                                    self.output_blocks[1:]):
#             x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
#             h = h + output_block(x, rbf, i, num_nodes=pos.size(0))
#         # Aggregate node (or atom) features into a molecule-level representation.
#         if batch is not None:
#             h = scatter(h, batch, dim=0, reduce='sum')
#         else:
#             h = h.sum(dim=0, keepdim=True)
#
#         # Process and fuse CLIP embeddings using cross-attention.
#         if clip_embeddings is not None:
#             B = h.size(0)
#             clip_embeddings = clip_embeddings.view(B, 768)
#             clip_feat = self.clip_head(clip_embeddings)  # [B, clip_encoder]
#             h = self.cross_attn_fusion(h, clip_feat)
#
#         # Final prediction.
#         out = self.lin_target(h)
#         return out

class GatedFusion(nn.Module):
    def __init__(self, mol_dim: int, clip_dim: int, fusion_dim: int):
        super().__init__()
        # A gate computed from the concatenation of both modalities.
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
        fused = g * mol_proj + (1 - g) * clip_proj  # Adaptive combination
        return fused


class DimeNetMulti(DimeNetPlusPlus):
    def __init__(
            self,
            hidden_channels: int,
            num_blocks: int,
            int_emb_size: int,
            basis_emb_size: int,
            out_emb_channels: int,
            num_spherical: int,
            num_radial: int,
            clip_encoder: int = 32,
            cutoff: float = 5.0,
            max_num_neighbors: int = 32,
            envelope_exponent: int = 5,
            num_before_skip: int = 1,
            num_after_skip: int = 2,
            num_output_layers: int = 3,
            act: Union[str, Callable] = 'swish',
            output_initializer: str = 'zeros',
    ):
        # Override original out_channels to produce an intermediate representation.
        super().__init__(
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,  # intermediate representation dimension
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
            output_initializer=output_initializer,
        )
        self.clip_encoder = clip_encoder

        # Process 512-dim CLIP embeddings.
        self.clip_head = Sequential(
            Linear(768, hidden_channels),
            ReLU(),
            Linear(hidden_channels, clip_encoder)
        )
        # Use gated fusion to combine molecular features (dimension = hidden_channels)
        # with the processed clip features (dimension = clip_encoder).
        # We'll set the fusion output dimension equal to hidden_channels.
        self.gated_fusion = GatedFusion(mol_dim=hidden_channels, clip_dim=clip_encoder, fusion_dim=hidden_channels)

        # Final prediction layer.
        self.lin_target = Linear(hidden_channels, 1)

    def forward(self, z: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None,
                clip_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional CLIP embeddings.

        Args:
            z (Tensor): Atomic numbers [num_atoms].
            pos (Tensor): Atomic positions [num_atoms, 3].
            batch (Tensor, optional): Batch indices for atoms.
            clip_embeddings (Tensor, optional): CLIP embeddings for each molecule [batch_size, 512].

        Returns:
            Tensor: Final target prediction [batch_size, 1].
        """
        # Create interaction graph.
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            edge_index, num_nodes=z.size(0))
        # Compute distances.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        # Compute angles (using the DimeNet++ variant formula).
        pos_jk, pos_ij = pos[idx_j] - pos[idx_k], pos[idx_i] - pos[idx_j]
        a = (pos_ij * pos_jk).sum(dim=-1)
        b = torch.cross(pos_ij, pos_jk, dim=1).norm(dim=-1)
        angle = torch.atan2(b, a)
        # Compute radial and spherical basis functions.
        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)
        # Embedding block.
        x = self.emb(z, rbf, i, j)
        # First output block produces an initial molecular representation.
        h = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))
        # Propagate through interaction and output blocks.
        for interaction_block, output_block in zip(self.interaction_blocks,
                                                   self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            h = h + output_block(x, rbf, i, num_nodes=pos.size(0))
        # Aggregate node (or atom) features into a molecule-level representation.
        if batch is not None:
            h = scatter(h, batch, dim=0, reduce='sum')
        else:
            h = h.sum(dim=0, keepdim=True)

        # Process CLIP embeddings using gated fusion.
        if clip_embeddings is not None:
            B = h.size(0)
            clip_embeddings = clip_embeddings.view(B, 768)
            clip_feat = self.clip_head(clip_embeddings)  # shape: [B, clip_encoder]
            h = self.gated_fusion(h, clip_feat)

        # Final prediction.
        out = self.lin_target(h)
        return out
