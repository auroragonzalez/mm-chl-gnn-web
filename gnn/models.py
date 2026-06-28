"""
GNN architectures for node-level chlorophyll-a regression.

Each model follows the same interface:
  - __init__(in_channels, hidden_channels, num_layers, dropout, ...)
  - forward(x, edge_index) -> Tensor of shape (N,)

Regression head: Linear -> ReLU -> Dropout -> Linear -> Softplus (pred > 0)

Implemented architectures:
  - GCN   (Kipf & Welling, 2017)
  - GAT   (Veličković et al., 2018)
  - GraphSAGE (Hamilton et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    SAGEConv,
    BatchNorm,
)


# ──────────────────────────────────────────────────────────────────────
# Base mixin
# ──────────────────────────────────────────────────────────────────────

class _GNNBase(nn.Module):
    """Shared utilities for all GNN regressors."""

    def reset_parameters(self):
        for module in self.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def enable_dropout_at_inference(self):
        """Force all Dropout layers to train mode (for MC Dropout UQ)."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    @staticmethod
    def _build_head(hidden_channels: int, dropout: float) -> nn.Sequential:
        """
        Regression head shared by all architectures.
        Linear -> ReLU -> Dropout -> Linear -> Softplus (ensures output > 0).
        """
        return nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
            nn.Softplus(),  # guarantees positive output
        )


# ──────────────────────────────────────────────────────────────────────
# GCN
# ──────────────────────────────────────────────────────────────────────

class GCNRegressor(_GNNBase):
    """
    Multi-layer GCN for node-level regression.

    Parameters
    ----------
    in_channels : int
        Number of input features per node.
    hidden_channels : int
        Hidden dimensionality.
    num_layers : int
        Number of GCN layers (>= 2).
    dropout : float
        Dropout probability applied after each hidden layer.
    residual : bool
        Add skip connections when hidden dim matches.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        residual: bool = True,
    ):
        super().__init__()
        self.dropout = dropout
        self.residual = residual

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer: in_channels -> hidden_channels
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.norms.append(BatchNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(BatchNorm(hidden_channels))

        # Last conv layer
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.norms.append(BatchNorm(hidden_channels))

        # Regression head
        self.head = self._build_head(hidden_channels, dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(x, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            # Skip connection for hidden layers (same dim)
            if self.residual and i > 0:
                h = h + x
            x = h
        return self.head(x).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────
# GAT
# ──────────────────────────────────────────────────────────────────────

class GATRegressor(_GNNBase):
    """
    Multi-layer GAT for node-level regression.

    Parameters
    ----------
    in_channels : int
        Number of input features per node.
    hidden_channels : int
        Hidden dimensionality **per head**.
    num_layers : int
        Number of GAT layers (>= 2).
    heads : int
        Number of attention heads (concatenated in hidden layers,
        averaged in the last layer).
    dropout : float
        Dropout probability.
    attn_dropout : float
        Dropout on attention coefficients.
    residual : bool
        Add skip connections.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
        attn_dropout: float = 0.1,
        residual: bool = True,
    ):
        super().__init__()
        self.dropout = dropout
        self.residual = residual

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads,
                    dropout=attn_dropout, concat=True)
        )
        self.norms.append(BatchNorm(hidden_channels * heads))

        # Hidden layers (input is hidden_channels * heads because of concat)
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels,
                        heads=heads, dropout=attn_dropout, concat=True)
            )
            self.norms.append(BatchNorm(hidden_channels * heads))

        # Last GAT layer: average heads instead of concatenating
        self.convs.append(
            GATConv(hidden_channels * heads, hidden_channels,
                    heads=1, dropout=attn_dropout, concat=False)
        )
        self.norms.append(BatchNorm(hidden_channels))

        # Projection for skip connections in hidden layers
        self.skip_proj = nn.Linear(in_channels, hidden_channels * heads, bias=False)

        # Regression head
        self.head = self._build_head(hidden_channels, dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x0 = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(x, edge_index)
            h = norm(h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            # Skip connections
            if self.residual and i == 0:
                h = h + self.skip_proj(x0)
            elif self.residual and 0 < i < len(self.convs) - 1 and h.shape == x.shape:
                h = h + x
            x = h
        return self.head(x).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────
# GraphSAGE
# ──────────────────────────────────────────────────────────────────────

class GraphSAGERegressor(_GNNBase):
    """
    Multi-layer GraphSAGE for node-level regression.

    Parameters
    ----------
    in_channels : int
        Number of input features per node.
    hidden_channels : int
        Hidden dimensionality.
    num_layers : int
        Number of SAGE layers (>= 2).
    dropout : float
        Dropout probability.
    aggr : str
        Aggregation type: ``"mean"`` or ``"max"``.
    residual : bool
        Add skip connections.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        aggr: str = "mean",
        residual: bool = True,
    ):
        super().__init__()
        self.dropout = dropout
        self.residual = residual

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        self.norms.append(BatchNorm(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.norms.append(BatchNorm(hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
        self.norms.append(BatchNorm(hidden_channels))

        # Regression head
        self.head = self._build_head(hidden_channels, dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(x, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.residual and i > 0:
                h = h + x
            x = h
        return self.head(x).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────

GNN_MODELS = {
    "GCN": GCNRegressor,
    "GAT": GATRegressor,
    "GraphSAGE": GraphSAGERegressor,
}
