"""ScanDDM gaze sampling with exact support for zero-priority pixels."""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from .pixel_race import race_DDM


class scanDDM(object):

    def __init__(self, fps, ffi=True, threshold=1.0, ndt=5, noise=0.1,
                 kappa=10, eta=7, device=None):
        self.DDM_sigma = noise
        self.k = kappa
        self.fps = fps
        self.threshold = threshold
        self.ndt = ndt
        self.eta = eta
        self.ffi = ffi
        self.device = device if device is not None else (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.whoami = 'ScanDDM'

    def get_map_from_binary(self, binmap):
        prior = np.asarray(binmap, dtype=np.float64)
        if prior.ndim != 2 or not np.isfinite(prior).all():
            raise ValueError('The gaze prior must be a finite 2D map')
        support = prior > 0
        if not np.any(support):
            raise ValueError('The gaze prior has no positive pixels to sample')

        smoothed = gaussian_filter(np.maximum(prior, 0), sigma=1 / 0.039)
        smoothed[~support] = 0
        peak = smoothed.max()
        if peak <= 0:
            raise ValueError('The gaze prior lost all positive pixels after smoothing')
        return smoothed / peak

    def _downsample_prior_map(self, saliency_map, factor=15):
        if isinstance(saliency_map, torch.Tensor):
            saliency_map = saliency_map.detach().cpu().numpy()
        source = np.asarray(saliency_map, dtype=np.float64)
        height, width = source.shape
        short_side = min(height, width)
        low_height = max(1, factor * height // short_side)
        low_width = max(1, factor * width // short_side)

        # Aggregate only original positive pixels. Interpolation would spread
        # probability into empty cells, and an epsilon would revive zero cells.
        ys, xs = np.nonzero(source > 0)
        cell_ids = (ys * low_height // height) * low_width + (xs * low_width // width)
        pooled = np.bincount(cell_ids, weights=source[ys, xs],
                             minlength=low_height * low_width)
        pooled = pooled.reshape(low_height, low_width)
        pooled /= pooled.max()
        return torch.from_numpy(pooled)

    @staticmethod
    def _sample_pixel_in_cell(prior_map, cell, low_shape):
        row, col = cell
        height, width = prior_map.shape
        low_height, low_width = low_shape
        y0 = (row * height + low_height - 1) // low_height
        y1 = ((row + 1) * height + low_height - 1) // low_height
        x0 = (col * width + low_width - 1) // low_width
        x1 = ((col + 1) * width + low_width - 1) // low_width
        patch = prior_map[y0:y1, x0:x1]
        ys, xs = np.nonzero(patch > 0)
        if len(ys) == 0:
            raise RuntimeError('The winning gaze cell has no positive-priority pixels')
        weights = patch[ys, xs]
        index = np.random.choice(len(ys), p=weights / weights.sum())
        return torch.tensor([x0 + int(xs[index]), y0 + int(ys[index])])

    def _simulate_next_foa(self, curr_fix, binary_map):
        prior_map = self.get_map_from_binary(binary_map)
        reduced_prior = self._downsample_prior_map(prior_map)
        low_height, low_width = reduced_prior.shape
        height, width = prior_map.shape
        current_cell = torch.tensor([
            min(int(curr_fix[1]) * low_height // height, low_height - 1),
            min(int(curr_fix[0]) * low_width // width, low_width - 1),
        ])
        race_model = race_DDM(
            fps=self.fps, downsampled_size=reduced_prior.shape,
            threshold=self.threshold, noise=self.DDM_sigma, kappa=self.k,
            eta=self.eta, ffi=self.ffi, device=self.device,
        )
        next_cell = (None, None)
        while next_cell == (None, None):
            next_cell, duration, _ = race_model.simulate_race(current_cell, reduced_prior)
        return self._sample_pixel_in_cell(prior_map, next_cell,
                                          reduced_prior.shape), duration
