import os
import json
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools import mask as mask_utils
from typing import List, Dict, Any, Callable, Optional
from scipy.ndimage import gaussian_filter
from .state import AgentState
from .rendering.viz import visualize

class ToolHandlers:
    ROLE_ORCHESTRATOR = 'orchestrator'
    ROLE_EXAMINE = 'examine'
    ROLE_JUDGE = 'judge'

    def __init__(self, state: AgentState, sam_service: Callable, llm_client: Callable, scanpath_model: Any, output_dirs: Dict[str, str], prompts: Dict[str, str], verbose: bool=True):
        self.state = state
        self.sam_service = sam_service
        self.llm_client = llm_client
        self.scanpath_model = scanpath_model
        self.output_dirs = output_dirs
        self.prompts = prompts
        self.verbose = verbose
        self._segmentation_round = 0
        self._total_masks_seen = 0
        self._judge_evidence = {}

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def handle_segment_phrase(self, tool_call: Dict[str, Any], messages: List[Dict]) -> None:
        self._log('\n🎯 Calling segment_phrase tool...')
        if 'text_prompts' not in tool_call['parameters']:
            raise ValueError(f"Missing 'text_prompts' parameter: {tool_call['parameters']}")
        text_prompts = tool_call['parameters']['text_prompts']
        print(f'Text prompts: {text_prompts}')
        if not isinstance(text_prompts, list):
            raise ValueError(f'text_prompts must be a list, got: {type(text_prompts)}')
        if not 1 <= len(text_prompts) <= 5:
            raise ValueError(f'text_prompts must contain 1-5 items, got {len(text_prompts)}')
        all_masks_data = []
        total_masks = 0
        for idx, prompt in enumerate(text_prompts):
            self._log(f"  Calling segmentation service for prompt {idx + 1}/{len(text_prompts)}: '{prompt}'")
            seg_output_path = self.sam_service(image_path=self.state.img_path, text_prompt=prompt, output_folder_path=self.output_dirs['sam'])
            with open(seg_output_path, 'r') as stream:
                seg_outputs = json.load(stream)
            num_masks = len(seg_outputs['pred_boxes'])
            self._log(f'    → Generated {num_masks} masks')
            all_masks_data.append({'prompt': prompt, 'output_path': seg_output_path, 'outputs': seg_outputs, 'num_masks': num_masks})
            total_masks += num_masks
        combined_outputs = self._merge_sam_outputs(all_masks_data)
        combined_output_path = os.path.join(self.output_dirs['sam'], f"combined_{'_'.join(text_prompts)[:50]}.json")
        with open(combined_output_path, 'w') as f:
            json.dump(combined_outputs, f)
        self.state.latest_sam_output_path = combined_output_path
        self.state.latest_sam_text_prompt = ', '.join(text_prompts)
        self.state.initialize_mask_scores(total_masks)
        self._segmentation_round += 1
        self._total_masks_seen += total_masks
        self._log(f'🎭 Initialized {total_masks} masks from {len(text_prompts)} prompts')
        messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': tool_call.get('full_text', '')}]})
        combined_vis_path = self._create_combined_visualization(all_masks_data)
        prompt_summary = ', '.join([f"'{d['prompt']}' ({d['num_masks']} masks)" for d in all_masks_data])
        sam_message = f'✅ SUCCESS: Generated {total_masks} masks from {len(text_prompts)} prompt(s): {prompt_summary}\n\nAll {total_masks} masks are shown below.\n\nState: `Masks_Ready`'
        self.state.curr_state = 'Masks_Ready'
        messages.append({'role': 'user', 'content': [{'type': 'text', 'text': sam_message}, {'type': 'image', 'image': combined_vis_path}]})
        self._log(f'\n>>> Segmentation output message:\n{sam_message}')
        self._log(f'\n>>> State:\n{self.state.curr_state}')

    def handle_predict_foa(self, tool_call: Dict[str, Any], messages: List[Dict]) -> None:
        self._log('\n👁️  Calling predict_foa tool (sample next fixation)...')
        with open(self.state.latest_sam_output_path, 'r') as stream:
            sam_outputs = json.load(stream)
        binary_masks = self._get_binary_masks(sam_outputs)
        prior_map = self._build_prior_map(binary_masks, self.state.mask_scores, ior=False, recent_penalty=0.8, smooth=False)
        self._log(f'Current mask_scores for Scanpath model: {self.state.mask_scores}')
        next_foa, prev_dur = self.scanpath_model._simulate_next_foa(self.state.curr_foa, prior_map)
        self.state.add_fixation(next_foa, prev_dur)
        chosen_idx = self._find_fixated_mask(next_foa, binary_masks)
        self._log(f'🎯 Fixation at ({next_foa[0]}, {next_foa[1]}) → Mask {chosen_idx + 1}')
        self.state.increment_fixation_count()
        if chosen_idx == self.state.last_predicted_mask_idx:
            self._log('Mask already examined. Staying in Masks_Ready.')
            messages.append({'role': 'user', 'content': [{'type': 'text', 'text': f'Mask {chosen_idx + 1} was already examined. State: `Masks_Ready`'}]})
            self.state.curr_state = 'Masks_Ready'
            self._log(f'\n>>> State:\n{self.state.curr_state}')
            self.state.update_mask_score(self.state.examined_masks_history[-1][0], self.state.examined_masks_history[-1][1])
            return
        self.state.last_predicted_mask_idx = chosen_idx
        foa_message = f'The predict_foa tool selected mask {chosen_idx + 1}. State: `Fixation_Set`.'
        self.state.curr_state = 'Fixation_Set'
        messages.append({'role': 'user', 'content': [{'type': 'text', 'text': foa_message}]})
        self._log(f'\n>>> predict_foa output:\n{foa_message}')
        self._log(f'\n>>> State:\n{self.state.curr_state}')

    def handle_examine_mask(self, tool_call: Dict[str, Any], messages: List[Dict]) -> None:
        self._log('\n🔍 Calling examine_mask tool...')
        with open(self.state.latest_sam_output_path, 'r') as stream:
            sam_outputs = json.load(stream)
        num_masks = len(sam_outputs['pred_masks'])
        mask_idx = tool_call['parameters'].get('mask_index')
        if mask_idx is None:
            mask_idx = self.state.last_predicted_mask_idx
        if mask_idx is None:
            raise ValueError('examine_mask called but no mask_index available')
        if not 0 <= mask_idx < num_masks:
            raise ValueError(f'Invalid mask index {mask_idx}')
        image_w_mask, image_w_zoom = visualize(sam_outputs, mask_idx)
        mask_path = os.path.join(self.output_dirs['sam'], f'{self.state.latest_sam_text_prompt}_mask_{mask_idx + 1}.png')
        zoom_path = os.path.join(self.output_dirs['sam'], f'{self.state.latest_sam_text_prompt}_mask_{mask_idx + 1}_zoom.png')
        image_w_mask.save(mask_path)
        image_w_zoom.save(zoom_path)
        examine_messages = [{'role': 'system', 'content': self.prompts['iterative_checking']}, {'role': 'user', 'content': [{'type': 'image', 'image': self.state.img_path}, {'type': 'text', 'text': f"User query: '{self.state.initial_text_prompt}'"}, {'type': 'image', 'image': mask_path}, {'type': 'image', 'image': zoom_path}]}]
        verdict_text = self.llm_client(examine_messages, agent_role=self.ROLE_EXAMINE)
        self._log(f'\n>>> examine_mask output:\n{verdict_text}')
        if verdict_text is None:
            raise ValueError('examine_mask: empty MLLM output')
        score = self._parse_verdict_score(verdict_text, examine_messages)
        self.state.update_mask_score(mask_idx, score)
        self._judge_evidence[(self._segmentation_round, mask_idx)] = (
            sam_outputs['pred_masks'][mask_idx], score)
        self._log(f'📊 Mask {mask_idx + 1} scored {score:.3f}')
        binary_masks = self._get_binary_masks(sam_outputs)
        messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': tool_call.get('full_text', '')}]})
        messages.append({'role': 'user', 'content': [{'type': 'text', 'text': f'Recorded score={score:.3f} for mask {mask_idx + 1}. State: `Mask_Examined`'}]})
        self.state.curr_state = 'Mask_Examined'
        self._log(f'\n>>> State:\n{self.state.curr_state}')

    def handle_judge(self, tool_call: Dict[str, Any], messages: List[Dict]) -> Dict[str, float]:
        """Ask the VLM for confidence using only masks examined so far."""
        if not self.state.latest_sam_output_path:
            raise ValueError('judge requires a previous segmentation')
        with open(self.state.latest_sam_output_path) as stream:
            sam_outputs = json.load(stream)
        height = int(sam_outputs['orig_img_h'])
        width = int(sam_outputs['orig_img_w'])
        prior_map = np.zeros((height, width), dtype=np.float64)
        for rle, score in self._judge_evidence.values():
            mask = mask_utils.decode({'size': (height, width), 'counts': rle})
            prior_map += np.clip(score, 0, 1) * mask
        prior_map = np.clip(prior_map, 0, 1)

        prior_path = os.path.join(
            self.output_dirs['main'], f'judge_prior_{self.state.generation_count}.png')
        Image.fromarray(np.rint(prior_map * 255).astype(np.uint8)).save(prior_path)
        judge_messages = [
            {'role': 'system', 'content': self.prompts['judge']},
            {'role': 'user', 'content': [
                {'type': 'image', 'image': self.state.img_path},
                {'type': 'text', 'text': f"User query: '{self.state.initial_text_prompt}'"},
                {'type': 'image', 'image': prior_path},
            ]},
        ]
        for attempt in range(2):
            confidence_text = self.llm_client(judge_messages, agent_role=self.ROLE_JUDGE)
            self._log(f'\n>>> judge output:\n{confidence_text}')
            try:
                confidence = self._parse_confidence_score(confidence_text)
                break
            except ValueError:
                if attempt:
                    raise
                judge_messages = judge_messages + [
                    {'role': 'assistant', 'content': confidence_text or ''},
                    {'role': 'user', 'content': 'Return a score from 0.00 to 1.00 inside '
                     '<confidence>...</confidence> tags.'},
                ]

        self.state.add_confidence_score(confidence)
        coverage = (len(self._judge_evidence) / self._total_masks_seen
                    if self._total_masks_seen else 0.0)
        self.state.coverage = coverage
        self.state.curr_state = 'Judge_Complete'
        messages.append({'role': 'assistant', 'content': [
            {'type': 'text', 'text': tool_call.get('full_text', '')}]})
        messages.append({'role': 'user', 'content': [
            {'type': 'text', 'text': f'Judge confidence={confidence:.3f}; '
             f'coverage={coverage:.3f}. The framework is processing the next step.'}]})
        self._log(f'Judge confidence: {confidence:.3f}; coverage: {coverage:.3f}')
        return {'confidence': confidence, 'coverage': coverage}

    @staticmethod
    def _parse_confidence_score(text: str | None) -> float:
        match = re.search(r'<confidence>\s*([^<]+?)\s*</confidence>', text or '')
        if match is None:
            raise ValueError('judge: invalid <confidence> score')
        try:
            confidence = float(match.group(1))
        except ValueError as error:
            raise ValueError('judge: invalid <confidence> score') from error
        if not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('judge: confidence must be in [0, 1]')
        return confidence

    def _get_binary_masks(self, sam_outputs: Dict) -> List[np.ndarray]:
        orig_h = int(sam_outputs['orig_img_h'])
        orig_w = int(sam_outputs['orig_img_w'])
        rle_masks = [{'size': (orig_h, orig_w), 'counts': rle} for rle in sam_outputs['pred_masks']]
        return [mask_utils.decode(rle) for rle in rle_masks]

    def _smooth_map(self, sal):
        sigma = 1 / 0.05
        Z = gaussian_filter(sal, sigma=sigma)
        Z = Z / np.max(Z)
        return Z

    def _build_prior_map(self, binary_masks: List[np.ndarray], mask_scores: Dict[int, float], smooth: bool=True, ior: bool=False, recent_penalty: float=0.75) -> np.ndarray:
        if not binary_masks:
            return np.zeros((720, 1280), dtype=np.float64)
        prior_map = np.zeros(binary_masks[0].shape, dtype=np.float64)
        last_fixated_mask = None
        if ior and self.state.last_predicted_mask_idx is not None:
            last_fixated_mask = self.state.last_predicted_mask_idx
        for idx, mask in enumerate(binary_masks):
            weight = mask_scores.get(idx, 0.0)
            if ior and idx == last_fixated_mask:
                weight *= recent_penalty
            prior_map += weight * mask.astype(np.float64)
        if smooth:
            prior_map = self._smooth_map(prior_map)
        return prior_map

    def _find_fixated_mask(self, fixation: torch.Tensor, binary_masks: List[np.ndarray]) -> int:
        x, y = (int(fixation[0]), int(fixation[1]))
        for idx, mask in enumerate(binary_masks):
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                if mask[y, x] > 0:
                    return idx
        self._log('⚠️  Fixation on empty region, finding nearest mask')
        centroids = []
        for mask in binary_masks:
            ys, xs = np.where(mask)
            if len(ys) > 0:
                centroids.append((ys.mean(), xs.mean()))
            else:
                centroids.append((np.nan, np.nan))
        distances = [np.hypot(c[0] - y, c[1] - x) if not np.isnan(c[0]) else np.inf for c in centroids]
        return int(np.argmin(distances))

    def _parse_verdict_score(self, text: str, messages: List[Dict]) -> float:
        try:
            extracted = text.split('<verdict>')[1].split('</verdict>')[0].strip()
            return float(extracted)
        except Exception:
            self._log('⚠️ Failed to parse verdict, retrying...')
            retry_text = self.llm_client(messages, agent_role=self.ROLE_EXAMINE)
            extracted = retry_text.split('<verdict>')[1].split('</verdict>')[0].strip()
            return float(extracted)

    def _merge_sam_outputs(self, all_masks_data: List[Dict]) -> Dict:
        if not all_masks_data:
            raise ValueError('No mask data to merge')
        template = all_masks_data[0]['outputs']
        combined = {'original_image_path': template['original_image_path'], 'orig_img_h': template['orig_img_h'], 'orig_img_w': template['orig_img_w'], 'pred_masks': [], 'pred_boxes': [], 'pred_scores': [], 'output_image_path': None}
        for data in all_masks_data:
            outputs = data['outputs']
            combined['pred_masks'].extend(outputs.get('pred_masks', []))
            combined['pred_boxes'].extend(outputs.get('pred_boxes', []))
            combined['pred_scores'].extend(outputs.get('pred_scores', []))
        return combined

    def _create_combined_visualization(self, all_masks_data: List[Dict]) -> str:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from PIL import Image
        import matplotlib.colors as mcolors
        orig_img = Image.open(self.state.img_path).convert('RGB')
        orig_img_array = np.array(orig_img)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(orig_img_array)
        colors = ['red', 'blue', 'green']
        mask_counter = 1
        legend_patches = []
        for prompt_idx, data in enumerate(all_masks_data):
            color = colors[prompt_idx % len(colors)]
            prompt = data['prompt']
            outputs = data['outputs']
            legend_patches.append(mpatches.Patch(color=color, label=f"'{prompt}' ({data['num_masks']} masks)"))
            orig_h = int(outputs['orig_img_h'])
            orig_w = int(outputs['orig_img_w'])
            rle_masks = [{'size': (orig_h, orig_w), 'counts': rle} for rle in outputs['pred_masks']]
            binary_masks = [mask_utils.decode(rle) for rle in rle_masks]
            for mask in binary_masks:
                colored_mask = np.zeros((*mask.shape, 4))
                rgba = mcolors.to_rgba(color)
                colored_mask[mask > 0] = [*rgba[:3], 0.4]
                ax.imshow(colored_mask)
                ys, xs = np.where(mask > 0)
                if len(ys) > 0:
                    center_y, center_x = (ys.mean(), xs.mean())
                    ax.text(center_x, center_y, str(mask_counter), color='white', fontsize=12, fontweight='bold', ha='center', va='center', bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
                mask_counter += 1
        ax.legend(handles=legend_patches, loc='upper right', fontsize=10)
        ax.set_title(f'Combined Masks from {len(all_masks_data)} Prompts', fontsize=14, fontweight='bold')
        ax.axis('off')
        output_path = os.path.join(self.output_dirs['sam'], f'combined_visualization_{self.state.generation_count}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
