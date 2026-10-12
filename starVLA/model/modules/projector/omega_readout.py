"""View-balanced state-conditioned queries with a spatial grid bypass."""

import torch
from torch import nn
from torch.nn import functional as F

from starVLA.model.modules.action_model.OmegaCrossAttentionHeader import CachedVisualAttention


class QueryReadBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = CachedVisualAttention(width, heads)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, query, memory):
        query = query + self.attention(self.query_norm(query), self.attention.prepare(self.memory_norm(memory)))
        return query + self.ffn(self.ffn_norm(query))


class StateQueryLocalReadout(nn.Module):
    def __init__(self, width=1024, state_dim=14, num_views=3, queries_per_view=64, local_grid=16, depth=2, heads=16):
        super().__init__()
        self.num_views = num_views
        self.local_grid = local_grid
        self.queries = nn.Parameter(torch.randn(num_views, queries_per_view, width) * 0.02)
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, width), nn.SiLU(), nn.Linear(width, width))
        nn.init.zeros_(self.state_encoder[-1].weight)
        nn.init.zeros_(self.state_encoder[-1].bias)
        self.blocks = nn.ModuleList([QueryReadBlock(width, heads) for _ in range(depth)])
        self.query_type = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.local_type = nn.Parameter(torch.randn(1, 1, width) * 0.02)

    def forward(self, features, state, grid_shape, keep_state=None):
        b, views, patches, width = features.shape
        height, grid_width = grid_shape
        if views != self.num_views or patches != height * grid_width:
            raise ValueError("Readout patch grid or number of cameras does not match")
        offset = self.state_encoder(state)
        if keep_state is not None:
            offset = offset * keep_state[:, None]
        query = self.queries[None].expand(b, -1, -1, -1) + offset[:, None, None]
        query = query.reshape(b * views, -1, width)
        memory = features.reshape(b * views, patches, width)
        for block in self.blocks:
            query = block(query, memory)
        query = query.reshape(b, -1, width) + self.query_type
        # Pool each camera independently; no learned global bottleneck on this path.
        local = memory.transpose(1, 2).reshape(b * views, width, height, grid_width)
        local = F.adaptive_avg_pool2d(local, (self.local_grid, self.local_grid))
        local = local.flatten(2).transpose(1, 2).reshape(b, -1, width) + self.local_type
        return torch.cat([query, local], dim=1)
