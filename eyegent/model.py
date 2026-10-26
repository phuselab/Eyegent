"""Single-image entry point; all transient agent inputs live in a temporary directory."""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .agent import run_agent
from .client import VLMClient
from .config import load_config
from .gaze import scanDDM
from .segmentation import CLIPSegConfig, CLIPSegService


@dataclass
class SimulationResult:
    scanpath: np.ndarray  # [N, 2], original returned coordinates
    durations: np.ndarray  # [N], seconds, original duration association
    stop_reason: str


class Eyegent:
    def __init__(self, config_path=None, *, config=None, llm_client=None,
                 segmenter=None, gaze=None, verbose=False):
        self.config = config if config is not None else load_config(config_path)
        self.llm_client = llm_client if llm_client is not None else VLMClient(self.config["llm"])
        self.segmenter = segmenter if segmenter is not None else CLIPSegService(
            CLIPSegConfig(**self.config["segmentation"])
        )
        self.gaze = gaze if gaze is not None else scanDDM(**self.config["scanpath"])
        self.verbose = verbose

    def run(self, image_path, query):
        image_path = Path(image_path).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must not be empty")
        with Image.open(image_path) as image:
            # The original Eyegent.run(size=img.size) receives (width, height).
            image_size = image.size
        with tempfile.TemporaryDirectory(prefix="eyegent-") as work_dir:
            scanpath, durations, reason = run_agent(
                str(image_path), query, self.config, self.llm_client,
                self.segmenter, self.gaze, work_dir, image_size, self.verbose,
            )
        coordinates = [point.detach().cpu().numpy() if hasattr(point, "detach")
                       else np.asarray(point) for point in scanpath]
        return SimulationResult(
            scanpath=np.asarray(coordinates).reshape(-1, 2),
            durations=np.asarray(durations, dtype=float), stop_reason=reason,
        )
