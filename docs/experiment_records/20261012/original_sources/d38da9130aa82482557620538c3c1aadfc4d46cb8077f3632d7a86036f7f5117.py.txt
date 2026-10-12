"""
adaptive_ensemble.py

"""

from collections import deque

import numpy as np


class AdaptiveEnsembler:
    def __init__(self, pred_action_horizon, adaptive_ensemble_alpha=0.0, stride=1):
        """
        Args:
            pred_action_horizon: max number of past chunk predictions to keep
                for ensembling.
            adaptive_ensemble_alpha: similarity-weighting sharpness.
            stride: number of physical steps between successive calls to
                `ensemble_action` (i.e. how many steps of each chunk are
                actually executed before the next replan). stride=1 matches
                the original per-step-replan behavior: each call ensembles a
                single action. stride>1 (receding-horizon / partial-chunk
                execution, e.g. predict 16 execute 8) ensembles a block of
                `stride` upcoming steps per call instead — a chunk that is
                `b` calls old contributes its entry at internal index
                `b * stride + j` for each of the `j in [0, stride)` steps
                about to be executed, and is skipped once that index runs
                past the chunk's length (it no longer covers that step).
        """
        self.pred_action_horizon = pred_action_horizon
        self.action_history = deque(maxlen=self.pred_action_horizon)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.stride = stride

    def reset(self):
        self.action_history.clear()

    def _weighted_blend(self, contribs):
        """contribs: (n, D) predictions for the same physical step, contribs[0] is the newest."""
        ref = contribs[0]
        dot_product = np.sum(contribs * ref, axis=1)
        norm_contribs = np.linalg.norm(contribs, axis=1)
        norm_ref = np.linalg.norm(ref)
        cos_similarity = dot_product / (norm_contribs * norm_ref + 1e-7)
        weights = np.exp(self.adaptive_ensemble_alpha * cos_similarity)
        weights = weights / weights.sum()
        return np.sum(weights[:, None] * contribs, axis=0)

    def ensemble_action(self, cur_action):
        # Store a copy: callers (e.g. in-place chunk mutation after this
        # call) must not be able to corrupt what's already recorded in history.
        self.action_history.append(cur_action.copy())

        if cur_action.ndim == 1:
            # Single-action-per-call path (stride=1 only): blend the current
            # action against the same-shaped entries in history.
            curr_act_preds = np.stack(self.action_history)
            return self._weighted_blend(curr_act_preds[::-1])

        # Chunked path: ensemble each of the next `stride` steps about to be
        # executed from the freshly predicted chunk against any older,
        # still-overlapping chunks (newest history entry = b=0 = cur_action).
        out = np.zeros((self.stride, cur_action.shape[1]), dtype=cur_action.dtype)
        history_newest_first = list(reversed(self.action_history))
        for j in range(self.stride):
            contribs = []
            for b, chunk in enumerate(history_newest_first):
                idx = b * self.stride + j
                if idx >= chunk.shape[0]:
                    break  # idx only grows with b, so older chunks won't cover it either
                contribs.append(chunk[idx])
            out[j] = self._weighted_blend(np.stack(contribs))

        return out
