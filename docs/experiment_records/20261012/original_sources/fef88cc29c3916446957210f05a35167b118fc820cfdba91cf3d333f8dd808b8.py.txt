from collections import deque

import torch
import torch.nn as nn


class LegacyCrossAttnPool(nn.Module):
    """Original single-shot cross-attention pooling (pre single-stage-pooling redesign).

    Learned query tokens cross-attend once over every backbone token; the
    attention output directly replaces the query (no residual, no norm). Kept
    verbatim so it can be selected as a baseline for A/B comparison against
    VisualMemoryPool.
    """

    def __init__(self, dim: int, num_query_tokens: int = 64, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, num_query_tokens, dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)

    def forward(self, kv: torch.Tensor) -> torch.Tensor:
        B = kv.shape[0]
        attn_dtype = self.cross_attn.in_proj_weight.dtype
        kv = kv.to(dtype=attn_dtype)
        query = self.query.expand(B, -1, -1).to(dtype=attn_dtype)
        attn_out, _ = self.cross_attn(query, kv, kv)
        return attn_out


class VisualMemoryPool(nn.Module):
    """Single-stage visual token pooling: cross-attention followed by self-attention.

    Learned memory (query) tokens cross-attend once over every backbone token
    (which carry view/spatial identity via VGGT's own 3D RoPE), then
    self-attend among themselves so pooled tokens can reconcile information
    gathered from different camera views before being handed to the action
    head. Plain single-shot cross-attention pooling has no mechanism for
    pooled tokens to interact with each other, so multi-view consistency is
    left entirely to the frozen backbone.
    """

    def __init__(
        self,
        dim: int,
        num_query_tokens: int = 64,
        num_heads: int = 8,
        num_self_attn_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, num_query_tokens, dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.cross_norm_q = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)

        self.self_norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(num_self_attn_layers))
        self.self_attn_layers = nn.ModuleList(
            nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
            for _ in range(num_self_attn_layers)
        )

    def forward(self, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            kv: (B, N, dim) backbone token features (live or cached).
        Returns:
            (B, num_query_tokens, dim) pooled condition tokens.
        """
        B = kv.shape[0]
        attn_dtype = self.cross_attn.in_proj_weight.dtype
        kv = kv.to(dtype=attn_dtype)
        query = self.query.expand(B, -1, -1).to(dtype=attn_dtype)

        attn_out, _ = self.cross_attn(self.cross_norm_q(query), kv, kv)
        query = query + attn_out

        for norm, self_attn in zip(self.self_norms, self.self_attn_layers):
            normed = norm(query)
            attn_out, _ = self_attn(normed, normed, normed)
            query = query + attn_out

        return query


class ResidualCrossAttnPool(nn.Module):
    """Cross-attention + residual + final norm, no self-attention.

    Step 1 of a staged re-introduction of VisualMemoryPool's self-attention
    reconciliation, after VisualMemoryPool (cross-attn + 2 self-attn,
    residual, no final norm) was measured to grow the residual stream ~3x in
    magnitude (127 -> 353) with nothing capping it before visual_proj, and
    to fail completely in closed-loop rollout despite converging fine
    offline — while LegacyCrossAttnPool (no residual, no norm at all)
    works. This isolates whether residual + a final norm alone (holding
    "depth" at effectively one attention op, same as LegacyCrossAttnPool)
    preserves stability. kv is normalized too (LegacyCrossAttnPool/
    VisualMemoryPool only normalized the query side), since the backbone's
    raw token scale isn't otherwise controlled going into attention.
    """

    def __init__(self, dim: int, num_query_tokens: int = 64, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, num_query_tokens, dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.cross_norm_q = nn.LayerNorm(dim)
        self.cross_norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, kv: torch.Tensor) -> torch.Tensor:
        B = kv.shape[0]
        attn_dtype = self.cross_attn.in_proj_weight.dtype
        kv = kv.to(dtype=attn_dtype)
        query = self.query.expand(B, -1, -1).to(dtype=attn_dtype)

        kv_normed = self.cross_norm_kv(kv)
        attn_out, _ = self.cross_attn(self.cross_norm_q(query), kv_normed, kv_normed)
        query = query + attn_out

        return self.final_norm(query)


class GatedMemoryPool(nn.Module):
    """Cross-attention + one LayerScale-gated self-attention layer + final norm.

    Step 2: reintroduces VisualMemoryPool's multi-view reconciliation
    self-attention layer on top of ResidualCrossAttnPool, but gates the
    self-attention residual update with a learnable per-channel scale
    (LayerScale, CaiT-style: "Going Deeper with Image Transformers"),
    initialized near zero. This lets the module start close to identity —
    self-attention contributes ~nothing initially — instead of forcing a
    randomly-initialized deep residual update to be absorbed at full
    strength from step 0, which is the standard fix for stabilizing added
    depth in a module trained from a cold start over a short budget.
    The cross-attention branch is left ungated: the learned query
    parameter's norm (~1) is negligible next to the cross-attention output's
    natural scale (~127, measured empirically), so a residual there behaves
    almost identically to LegacyCrossAttnPool's direct (non-residual) output
    regardless of gating.
    """

    def __init__(
        self,
        dim: int,
        num_query_tokens: int = 64,
        num_heads: int = 8,
        layerscale_init: float = 1e-4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, num_query_tokens, dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.cross_norm_q = nn.LayerNorm(dim)
        self.cross_norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)

        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.self_gamma = nn.Parameter(layerscale_init * torch.ones(dim))

        self.final_norm = nn.LayerNorm(dim)

    def forward(self, kv: torch.Tensor) -> torch.Tensor:
        B = kv.shape[0]
        attn_dtype = self.cross_attn.in_proj_weight.dtype
        kv = kv.to(dtype=attn_dtype)
        query = self.query.expand(B, -1, -1).to(dtype=attn_dtype)

        kv_normed = self.cross_norm_kv(kv)
        attn_out, _ = self.cross_attn(self.cross_norm_q(query), kv_normed, kv_normed)
        query = query + attn_out

        normed = self.self_norm(query)
        attn_out, _ = self.self_attn(normed, normed, normed)
        query = query + self.self_gamma * attn_out

        return self.final_norm(query)


class GatedRecurrentMemoryPool(nn.Module):
    """Persistent memory updated by a gated convex combination, not residual addition.

    StreamingMemoryPool's failure mode was `memory = memory + attn_out`,
    repeated once per timestep, with no gate and no norm in between — the
    same unbounded-residual-accumulation problem later diagnosed in
    VisualMemoryPool (there: a single flat call; here: the same thing
    compounded recurrently across timesteps), except never fixed. This
    module replaces that update rule with a GRU-style gated convex
    combination:

        candidate = CrossAttn(LN(memory), LN(new_kv))
        z = sigmoid(Gate([memory, candidate]))
        memory = (1 - z) * memory + z * candidate

    A convex combination is bounded by construction — regardless of how many
    recurrent steps are taken, memory can never leave the range spanned by
    its own history and incoming candidates. A gated *residual* (even
    LayerScale-scaled) can still drift over a long streaming horizon since
    small updates keep summing; this cannot, since old and new are always
    interpolated, never added. A LayerNorm follows every update (not just
    once at the end of a fixed window, as in GatedMemoryPool), since with
    genuine streaming there is no fixed "end" for a norm to wait for.

    Self-attention reconciliation among memory tokens (LayerScale-gated, as
    in GatedMemoryPool — validated to do real reconciliation work, not just
    be tolerated) follows each update, also norm-capped.

    Two entry points, matching StreamingMemoryPool's pattern:
      - `step(kv_step)`: advance persistent state by one new observation.
      - `forward_sequence(kv_steps)`: resets and replays `step()` over a
        fixed window at once (oldest-to-newest) — used for training against
        the existing fixed [0,-8,-16]-style window without any data
        pipeline changes, and as a "replay mode" A/B against flat pooling
        before committing to real incremental deployment.
    """

    def __init__(
        self,
        dim: int,
        num_query_tokens: int = 64,
        num_heads: int = 8,
        layerscale_init: float = 1e-4,
        dropout: float = 0.0,
        window_steps: int = 3,
    ):
        super().__init__()
        # Not used internally (the recurrence itself has no fixed window —
        # step()/forward_sequence accept any number of timesteps). Stored
        # only so callers (VGGTOmegaOFT._pool_visual) can chunk a flattened
        # kv tensor into per-timestep pieces the same way they do for
        # StreamingMemoryPool, without special-casing this class.
        self.window_steps = window_steps

        self.query = nn.Parameter(torch.zeros(1, num_query_tokens, dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.cross_norm_q = nn.LayerNorm(dim)
        self.cross_norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.update_gate = nn.Linear(2 * dim, dim)
        self.update_norm = nn.LayerNorm(dim)

        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.self_gamma = nn.Parameter(layerscale_init * torch.ones(dim))
        self.self_norm_out = nn.LayerNorm(dim)

        self._memory: torch.Tensor | None = None

    def reset(self) -> None:
        self._memory = None

    def step(self, kv_step: torch.Tensor) -> torch.Tensor:
        """
        Advance persistent memory by one new observation.

        Args:
            kv_step: (B, n, dim) backbone tokens for the single newest
                observation (all camera views at that step, concatenated).
        Returns:
            (B, num_query_tokens, dim) updated pooled memory tokens.
        """
        B = kv_step.shape[0]
        attn_dtype = self.cross_attn.in_proj_weight.dtype
        kv_step = kv_step.to(dtype=attn_dtype)

        if self._memory is None:
            self._memory = self.query.expand(B, -1, -1).to(dtype=attn_dtype)
        memory = self._memory

        kv_normed = self.cross_norm_kv(kv_step)
        candidate, _ = self.cross_attn(self.cross_norm_q(memory), kv_normed, kv_normed)

        z = torch.sigmoid(self.update_gate(torch.cat([memory, candidate], dim=-1)))
        memory = (1 - z) * memory + z * candidate
        memory = self.update_norm(memory)

        normed = self.self_norm(memory)
        self_out, _ = self.self_attn(normed, normed, normed)
        memory = memory + self.self_gamma * self_out
        memory = self.self_norm_out(memory)

        self._memory = memory
        return memory

    def forward_sequence(self, kv_steps) -> torch.Tensor:
        """
        Reset and replay the recurrence over a full window at once.

        Args:
            kv_steps: (B, T, n, dim) tensor, or list[T] of (B, n, dim)
                tensors, ordered most-recent-first (index 0 = current step) —
                same convention as StreamingMemoryPool.forward_sequence.
        Returns:
            (B, num_query_tokens, dim) pooled memory tokens after seeing the
            whole window, gradient-connected through every step.
        """
        if torch.is_tensor(kv_steps):
            kv_steps = [kv_steps[:, t] for t in range(kv_steps.shape[1])]

        self.reset()
        memory = None
        for kv in reversed(kv_steps):  # oldest -> newest, ends on the current frame
            memory = self.step(kv)
        return memory


class HierarchicalGatedMemoryPool(nn.Module):
    """Two-level memory: per-chunk flat pooling feeding a cross-chunk gated recurrence.

    Motivated by a structural gap in every fixed-window design used so far
    (VisualMemoryPool, GatedMemoryPool, GatedRecurrentMemoryPool): no matter
    how long an episode runs, they only ever see a fixed 16-real-step
    lookback ([0,-8,-16]), so they have zero representational capacity for
    anything that happened earlier in a long, multi-stage task. This module
    lets a persistent global memory accumulate across an entire episode.

    Level 1 (chunk): one execution window's worth of frames (e.g. 8
    timesteps x 3 views, matching execute_horizon=8) goes through ONE joint
    VGGT-Omega backbone pass (real cross-frame/cross-view attention within
    the chunk), then a GatedMemoryPool-style flat pool (cross-attn + residual
    + final norm + one LayerScale-gated self-attention layer) compresses it
    to a 64-token chunk summary. Unlike hist8stride1_gated's failed
    experiment (8 consecutive frames fed as a one-shot replacement
    observation, near-duplicates with no temporal diversity, 0% rollout SR),
    the within-chunk redundancy here isn't asked to carry temporal signal —
    it plausibly helps as an averaging/denoising effect over close-together
    views of the same moment.

    Level 2 (global): each chunk summary updates a persistent global memory
    via GatedRecurrentMemoryPool's bounded gated convex-combination update
    (memory = (1-z)*memory + z*candidate), carried across as many chunks as
    the episode has — not capped at a fixed window. Temporal diversity comes
    from the sequence of per-chunk summaries (chunk@t=0, chunk@t=8, ...),
    the same "genuinely different snapshot per decision cycle" reasoning
    that motivated the original wide-stride [0,-8,-16] design, just now
    unbounded in how far back it can reach.

    Offline-validated: global memory norm stays exactly bounded (1.000x
    range) over 25 chunks (~200 frames, the upper end of the target task
    length), vs. 24.5x unbounded growth for naive residual accumulation fed
    the same chunk summaries — see validate_hierarchical_memory.py.

    This class only composes GatedMemoryPool + GatedRecurrentMemoryPool
    (both already independently validated) — no new attention/gating math.
    """

    def __init__(
        self,
        dim: int,
        num_query_tokens: int = 64,
        num_heads: int = 8,
        layerscale_init: float = 1e-4,
        dropout: float = 0.0,
        num_chunks: int = 3,
    ):
        super().__init__()
        # Not used internally by the recurrence (which accepts any number of
        # chunks); stored so VGGTOmegaOFT can split the flattened per-chunk
        # image tensor into this many separate backbone passes.
        self.num_chunks = num_chunks

        self.chunk_pool = GatedMemoryPool(
            dim=dim, num_query_tokens=num_query_tokens, num_heads=num_heads,
            layerscale_init=layerscale_init, dropout=dropout,
        )
        self.global_pool = GatedRecurrentMemoryPool(
            dim=dim, num_query_tokens=num_query_tokens, num_heads=num_heads,
            layerscale_init=layerscale_init, dropout=dropout,
        )

    def reset(self) -> None:
        self.global_pool.reset()

    def step(self, chunk_kv: torch.Tensor) -> torch.Tensor:
        """
        Advance global memory by one new chunk.

        Args:
            chunk_kv: (B, n, dim) raw backbone tokens for one chunk (all
                timesteps/views in that chunk's joint VGGT pass, concatenated).
        Returns:
            (B, num_query_tokens, dim) updated global memory tokens.
        """
        chunk_summary = self.chunk_pool(chunk_kv)       # level 1: flat pool this chunk
        return self.global_pool.step(chunk_summary)      # level 2: fold into global memory

    def forward_sequence(self, chunk_kvs) -> torch.Tensor:
        """
        Reset and replay the recurrence over a full sequence of chunks.

        Args:
            chunk_kvs: list[num_chunks] of (B, n, dim) tensors, ordered
                most-recent-first (index 0 = current chunk) — same
                convention as GatedRecurrentMemoryPool.forward_sequence.
        Returns:
            (B, num_query_tokens, dim) global memory after seeing every
            chunk, gradient-connected through both levels at every step.
        """
        self.reset()
        memory = None
        for chunk_kv in reversed(chunk_kvs):  # oldest -> newest
            memory = self.step(chunk_kv)
        return memory


class StreamingMemoryPool(nn.Module):
    """Recurrent visual token pooling over a bounded sliding window of timesteps.

    VisualMemoryPool cross-attends once over every backbone token from every
    timestep flattened together, so a 9-timestep history costs the same
    single non-causal attention as a 1-timestep one and the pooled tokens
    have no notion of *when* a piece of visual evidence came from. This
    module instead treats the history window as a sequence: persistent
    memory tokens are updated one timestep at a time, at each step
    cross-attending only over a bounded cache of the last `window_steps`
    timesteps' tokens (oldest evicted once the cache is full) plus a
    self-attention refinement, then carried forward as the query for the
    next timestep. A learned per-offset embedding tags each cached
    timestep's tokens with how many steps back it is, since that
    information is otherwise lost once the window starts sliding.

    Two entry points share one underlying recurrence:
      - `step(kv_step)`: advance persistent state by exactly one timestep.
        Used for true incremental streaming, where a caller feeds one new
        timestep's tokens per call across a whole episode.
      - `forward_sequence(kv_steps)`: resets state, then replays `step()`
        oldest-to-newest over a full history window given at once. This is
        what both training (unrolling over the fixed history window in a
        training example) and the current eval protocol (which resends the
        full window every replan cycle, not a single incremental frame) use
        — it keeps the training-time computation graph identical to
        step-by-step streaming inference, so there's no train/inference
        mismatch even though the eval harness doesn't do incremental
        transmission yet.
    """

    def __init__(
        self,
        dim: int,
        num_query_tokens: int = 64,
        num_heads: int = 8,
        num_self_attn_layers: int = 2,
        window_steps: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.window_steps = window_steps
        self.query = nn.Parameter(torch.zeros(1, num_query_tokens, dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.cross_norm_q = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)

        self.self_norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(num_self_attn_layers))
        self.self_attn_layers = nn.ModuleList(
            nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
            for _ in range(num_self_attn_layers)
        )

        # step_pos_emb[0] tags the newest cached timestep, step_pos_emb[1]
        # one step older, etc. — added to that timestep's tokens before
        # cross-attention so recency survives the window sliding.
        self.step_pos_emb = nn.Parameter(torch.zeros(window_steps, dim))
        nn.init.trunc_normal_(self.step_pos_emb, std=0.02)

        self._memory = None                       # (B, num_query_tokens, dim), None until first step()
        self._window = deque(maxlen=window_steps)  # cached per-timestep kv, oldest-first

    def reset(self) -> None:
        self._memory = None
        self._window.clear()

    def _cross_and_self(self, memory: torch.Tensor, kv_window: torch.Tensor) -> torch.Tensor:
        attn_dtype = self.cross_attn.in_proj_weight.dtype
        kv_window = kv_window.to(dtype=attn_dtype)
        memory = memory.to(dtype=attn_dtype)

        attn_out, _ = self.cross_attn(self.cross_norm_q(memory), kv_window, kv_window)
        memory = memory + attn_out

        for norm, self_attn in zip(self.self_norms, self.self_attn_layers):
            normed = norm(memory)
            attn_out, _ = self_attn(normed, normed, normed)
            memory = memory + attn_out

        return memory

    def step(self, kv_step: torch.Tensor) -> torch.Tensor:
        """
        Advance persistent state by one timestep.

        Args:
            kv_step: (B, n, dim) backbone tokens for a single newest
                timestep (all camera views at that timestep already
                concatenated along the token dim).
        Returns:
            (B, num_query_tokens, dim) updated pooled memory tokens.
        """
        B = kv_step.shape[0]
        if self._memory is None:
            attn_dtype = self.cross_attn.in_proj_weight.dtype
            self._memory = self.query.expand(B, -1, -1).to(dtype=attn_dtype)

        self._window.append(kv_step)
        # reversed(): newest-first, so offset 0 always tags the timestep
        # just appended regardless of how full the window is.
        chunks = [
            kv + self.step_pos_emb[offset].to(dtype=kv.dtype)
            for offset, kv in enumerate(reversed(self._window))
        ]
        kv_window = torch.cat(chunks, dim=1)

        self._memory = self._cross_and_self(self._memory, kv_window)
        return self._memory

    def forward_sequence(self, kv_steps) -> torch.Tensor:
        """
        Reset and replay the recurrence over a full history window at once.

        Args:
            kv_steps: (B, T, n, dim) tensor, or list[T] of (B, n, dim)
                tensors — backbone tokens per timestep, ordered
                most-recent-first (index 0 = current step), matching
                `video_observation_indices` / the time-major flattened
                image order the model already assumes.
        Returns:
            (B, num_query_tokens, dim) pooled memory tokens after seeing the
            whole window, gradient-connected through every step.
        """
        if torch.is_tensor(kv_steps):
            kv_steps = [kv_steps[:, t] for t in range(kv_steps.shape[1])]

        self.reset()
        memory = None
        for kv in reversed(kv_steps):  # oldest -> newest, ends on the current frame
            memory = self.step(kv)
        return memory
