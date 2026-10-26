from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from scipy.ndimage import gaussian_filter

def _load_clipseg():
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    model_name = 'CIDAS/clipseg-rd64-refined'
    processor = CLIPSegProcessor.from_pretrained(model_name, use_fast=False)
    model = CLIPSegForImageSegmentation.from_pretrained(model_name)
    model.eval()
    return (processor, model)

@dataclass
class CLIPSegConfig:
    prob_threshold: float = 0.4
    min_area_ratio: float = 0.001
    use_ellipse: bool = True
    max_masks_per_prompt: int = 10
    device: str = 'cuda'

def _smooth_map(sal):
    sigma = 1 / 0.039
    Z = gaussian_filter(sal, sigma=sigma)
    Z = Z / np.max(Z)
    return Z

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _logits_to_prob_map(logits: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    probs = _sigmoid(logits).astype(np.float32)
    for _ in range(40):
        probs = np.clip(probs - probs.mean(), 0, np.inf)
    probs /= np.max(probs)
    if probs.shape != (target_h, target_w):
        probs = cv2.resize(probs, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return probs

def _prob_map_to_binary(prob_map: np.ndarray, threshold: float) -> np.ndarray:
    return (prob_map >= threshold).astype(np.uint8)

def _fit_ellipse_mask(component_mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return component_mask
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return component_mask
    try:
        ellipse = cv2.fitEllipse(largest)
    except cv2.error:
        return component_mask
    ellipse_mask = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(ellipse_mask, ellipse, color=1, thickness=-1)
    return ellipse_mask

def _connected_component_masks(binary_map: np.ndarray, min_area_ratio: float, use_ellipse: bool, max_masks: int) -> List[np.ndarray]:
    h, w = binary_map.shape
    total_pixels = h * w
    min_area = max(1, int(min_area_ratio * total_pixels))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
    masks: List[Tuple[int, np.ndarray]] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = (labels == label).astype(np.uint8)
        if use_ellipse:
            component = _fit_ellipse_mask(component, (h, w))
            area = int(component.sum())
        masks.append((area, component))
    masks.sort(key=lambda t: t[0], reverse=True)
    return [m for _, m in masks[:max_masks]]

def _mask_to_rle(mask: np.ndarray) -> str:
    rle = mask_utils.encode(np.asfortranarray(mask))
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode('utf-8')
    return rle['counts']

def _mask_to_box(mask: np.ndarray) -> List[float]:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

def _mask_score(mask: np.ndarray, prob_map: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0.0
    return float(prob_map[ys, xs].mean())

class CLIPSegService:

    def __init__(self, config: Optional[CLIPSegConfig]=None):
        self.config = config or CLIPSegConfig()
        self._processor = None
        self._model = None

    def _ensure_model(self):
        if self._processor is None:
            import torch
            self._processor, self._model = _load_clipseg()
            self._model = self._model.to(self.config.device)

    def _run_clipseg(self, image: Image.Image, text_prompt: str) -> np.ndarray:
        import torch
        inputs = self._processor(text=[text_prompt], images=[image], return_tensors='pt', padding=True)
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        logits = outputs.logits[0].cpu().numpy()
        return logits

    def __call__(self, image_path: str, text_prompt: str, output_folder_path: str) -> str:
        self._ensure_model()
        os.makedirs(output_folder_path, exist_ok=True)
        image = Image.open(image_path).convert('RGB')
        orig_w, orig_h = image.size
        logits = self._run_clipseg(image, text_prompt)
        prob_map = _logits_to_prob_map(logits, orig_h, orig_w)
        binary_map = _prob_map_to_binary(prob_map, self.config.prob_threshold)
        component_masks = _connected_component_masks(binary_map, min_area_ratio=self.config.min_area_ratio, use_ellipse=self.config.use_ellipse, max_masks=self.config.max_masks_per_prompt)
        pred_masks: List[str] = []
        pred_boxes: List[List[float]] = []
        pred_scores: List[float] = []
        for mask in component_masks:
            pred_masks.append(_mask_to_rle(mask))
            pred_boxes.append(_mask_to_box(mask))
            pred_scores.append(_mask_score(mask, prob_map))
        safe_prompt = text_prompt.replace(' ', '_').replace('/', '-')[:60]
        output_filename = f'clipseg_{safe_prompt}.json'
        output_path = os.path.join(output_folder_path, output_filename)
        result = {'original_image_path': str(image_path), 'orig_img_h': orig_h, 'orig_img_w': orig_w, 'pred_masks': pred_masks, 'pred_boxes': pred_boxes, 'pred_scores': pred_scores, 'output_image_path': None, 'segmentation_backend': 'clipseg', 'text_prompt': text_prompt, 'prob_threshold': self.config.prob_threshold, 'use_ellipse': self.config.use_ellipse}
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        return output_path
