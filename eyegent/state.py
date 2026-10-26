import torch
import numpy as np
from typing import List, Dict, Set, Tuple, Optional

class AgentState:

    def __init__(self, config: dict, paths: dict, img_path: str, initial_text_prompt: str, img_size: Optional[Tuple[int, int]]=None):
        self.config = config
        self.paths = paths
        self.img_path = img_path
        self.initial_text_prompt = initial_text_prompt
        self.curr_state = 'Initial'
        self.generation_count = 0
        self.fixation_count = 0
        self.latest_sam_output_path = ''
        self.latest_sam_text_prompt = ''
        self.used_text_prompts: Set[str] = set()
        self.mask_scores: Dict[int, float] = {}
        self.last_predicted_mask_idx: Optional[int] = None
        initial_foa = self._get_initial_fixation(img_size)
        self.scanpath: List[torch.Tensor] = [initial_foa]
        self.durs: List[float] = []
        self.curr_foa = initial_foa
        self.curr_dir = 0
        self.confidence_scores: List[float] = []
        self.examined_masks_history: List[Tuple[int, float]] = []
        self.mask_scores_history: List[Dict[int, float]] = []
        self.coverage = 0

    def _get_initial_fixation(self, img_size: Optional[Tuple[int, int]]) -> torch.Tensor:
        if img_size is None:
            return torch.tensor([360, 640])
        return torch.tensor([int(img_size[0] // 2), int(img_size[1] // 2)])

    def add_used_prompt(self, prompt: str) -> None:
        self.used_text_prompts.add(prompt)
        self.latest_sam_text_prompt = prompt

    def is_prompt_used(self, prompt: str) -> bool:
        return prompt in self.used_text_prompts

    def get_previously_used_prompts(self) -> List[str]:
        if self.latest_sam_text_prompt:
            return sorted(p for p in self.used_text_prompts if p != self.latest_sam_text_prompt)
        return sorted(self.used_text_prompts)

    def initialize_mask_scores(self, num_masks: int) -> None:
        self.mask_scores = {i: 1.0 / num_masks for i in range(num_masks)}
        self.last_predicted_mask_idx = None

    def update_mask_score(self, mask_idx: int, score: float) -> None:
        self.mask_scores[mask_idx] = score
        self.examined_masks_history.append((mask_idx, score))
        self.mask_scores_history.append(dict(self.mask_scores))

    def add_fixation(self, foa: torch.Tensor, duration: float) -> None:
        self.scanpath.append(foa)
        self.durs.append(duration)
        self.curr_foa = foa

    def increment_fixation_count(self) -> None:
        self.fixation_count += 1

    def increment_generation_count(self) -> None:
        self.generation_count += 1

    def add_confidence_score(self, score: float) -> None:
        self.confidence_scores.append(score)

    def get_unexamined_count(self) -> int:
        if not self.mask_scores:
            return 0
        initial_score = 1.0 / len(self.mask_scores)
        return sum((1 for score in self.mask_scores.values() if abs(score - initial_score) < 1e-06))

    def get_examined_count(self) -> int:
        return len(self.mask_scores) - self.get_unexamined_count()

    def get_total_duration(self) -> float:
        return sum(self.durs)
