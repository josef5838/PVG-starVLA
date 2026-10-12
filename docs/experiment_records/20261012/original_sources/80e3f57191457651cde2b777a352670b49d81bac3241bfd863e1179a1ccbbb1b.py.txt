"""
Episode-ordered chunk streaming for Option B (true stateful) training of
HierarchicalGatedMemoryPool's global memory.

Unlike the standard i.i.d. random-window dataset (each training example is
an independent, randomly-sampled fixed window), this delivers TEMPORALLY
ORDERED 8-timestep chunks, walking each of `num_slots` parallel "episode
streams" from the start of a randomly-chosen episode to its end, then
starting a new episode in that slot. The training loop carries a persistent
per-slot memory tensor across calls, exactly mirroring what happens at real
deployment (one new chunk arrives, updates the running memory, repeat).

Deliberately NOT a torch.utils.data.Dataset/DataLoader: state must advance
in a strict, non-shuffled order per slot, which multi-worker DataLoader
parallelism can't safely guarantee. Runs single-process; each next_batch()
call reuses LeRobotSingleDataset's own get_step_data/transforms/_pack_sample
(already-correct, already-tested primitives) for the actual frame/action
fetching — only the "which (trajectory, position) comes next, and when to
reset" scheduling is new.
"""

import random


class EpisodeChunkStream:
    def __init__(self, single_dataset, num_slots: int, chunk_timesteps: int = 8, seed: int = 0):
        """
        Args:
            single_dataset: a LeRobotSingleDataset built with a data_cfg whose
                video_observation_indices cover exactly `chunk_timesteps`
                consecutive steps, most-recent-first (e.g.
                AgilexDataConfigHist8Stride1: [0,-1,...,-7]) — get_step_data
                then returns exactly one chunk's worth of frames per call.
            num_slots: number of parallel, independently-progressing episode
                streams (this run's effective batch size).
            chunk_timesteps: must match len(video_observation_indices) on
                single_dataset's data_cfg.
        """
        self.dataset = single_dataset
        self.num_slots = num_slots
        self.chunk_timesteps = chunk_timesteps
        self.rng = random.Random(seed)

        self.trajectory_ids = list(single_dataset.trajectory_ids)
        self.trajectory_lengths = dict(zip(single_dataset.trajectory_ids, single_dataset.trajectory_lengths))
        if not self.trajectory_ids:
            raise ValueError("Dataset has no trajectories to stream from.")
        too_short = [tid for tid in self.trajectory_ids if self.trajectory_lengths[tid] < chunk_timesteps]
        if too_short:
            raise ValueError(
                f"{len(too_short)} trajectories are shorter than chunk_timesteps={chunk_timesteps}; "
                f"can't produce even one full chunk from them."
            )

        # Per-slot state. _slot_next_end[i] is the base_index (absolute step,
        # most-recent step in the chunk) to fetch next for slot i.
        self._slot_traj = [None] * num_slots
        self._slot_next_end = [None] * num_slots
        for slot in range(num_slots):
            self._start_new_episode(slot)

    def _start_new_episode(self, slot: int) -> None:
        self._slot_traj[slot] = self.rng.choice(self.trajectory_ids)
        # First chunk covers absolute steps [0..chunk_timesteps-1]; its
        # "end" (base_index, most-recent step) is chunk_timesteps - 1.
        self._slot_next_end[slot] = self.chunk_timesteps - 1

    def __len__(self) -> int:
        """Rough proxy for _log_metrics' epoch calculation, not exact."""
        total_steps = sum(self.trajectory_lengths.values())
        return max(total_steps // (self.num_slots * self.chunk_timesteps), 1)

    def next_batch(self):
        """
        Returns:
            examples: list[num_slots] of packed sample dicts (same shape as
                _pack_sample's output: "image", "state", "action", "lang").
            reset_mask: list[bool], True for slots whose returned chunk is
                the FIRST chunk of a (possibly brand-new) episode this call —
                the caller must reinitialize that slot's carried memory from
                the model's learned initial query rather than reusing the
                previous (unrelated) episode's leftover memory.
        """
        examples = []
        reset_mask = []
        for slot in range(self.num_slots):
            tid = self._slot_traj[slot]
            end = self._slot_next_end[slot]
            is_reset = end == self.chunk_timesteps - 1

            raw = self.dataset.get_step_data(tid, end)
            data = self.dataset.transforms(raw)
            sample = self.dataset._pack_sample(data)

            examples.append(sample)
            reset_mask.append(is_reset)

            next_end = end + self.chunk_timesteps
            if next_end >= self.trajectory_lengths[tid]:
                # Episode exhausted (any remainder < chunk_timesteps steps at
                # the tail is dropped). Queue a fresh episode for next call;
                # this call's data is still valid, from the current episode.
                self._start_new_episode(slot)
            else:
                self._slot_next_end[slot] = next_end

        return examples, reset_mask
