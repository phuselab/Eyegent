"""Agent-controlled loop with semantic judge and simulated-time limits."""

import os
from pathlib import Path

import numpy as np

from .messages import (
    create_initial_messages,
    parse_tool_call_with_retry,
    prune_messages_for_next_round,
)
from .state import AgentState
from .tools import ToolHandlers


def load_prompts(judge_threshold=None):
    directory = Path(__file__).parent / "prompts"
    names = {
        "system": "AV_new_system_prompt_v2.txt",
        "iterative_checking": "AV_new_system_prompt_iterative_checking.txt",
        "judge": "AV_judge_prompt.txt",
    }
    prompts = {key: (directory / name).read_text().strip() for key, name in names.items()}
    if judge_threshold is not None:
        start = prompts["system"].find("### State: `Judge_Complete`")
        end = prompts["system"].find("\n---", start)
        if start < 0 or end < 0:
            raise RuntimeError("The orchestrator prompt's judge state has changed")
        runtime_rule = (
            "### State: `Judge_Complete`\n\n"
            "The framework handles this state without another orchestrator reply. "
            f"If judge confidence ≥ {judge_threshold:g}, the workflow terminates. "
            "Otherwise, the framework samples the next fixation, subject to the "
            "duration and generation limits. If the loop continues, follow the "
            "new `Fixation_Set` or `Masks_Ready` state in the latest tool result."
        )
        prompts["system"] = (prompts["system"][:start] + runtime_rule
                             + prompts["system"][end:])
    return prompts


def sample_final_fixation(state, gaze):
    """Retain the original final sample and its duration association."""
    import json
    from pycocotools import mask as mask_utils

    with open(state.latest_sam_output_path) as stream:
        outputs = json.load(stream)
    height, width = int(outputs["orig_img_h"]), int(outputs["orig_img_w"])
    masks = [mask_utils.decode({"size": (height, width), "counts": rle})
             for rle in outputs["pred_masks"]]
    prior = np.zeros((height, width), dtype=np.float64)
    for index, mask in enumerate(masks):
        prior += state.mask_scores[index] * mask.astype(np.float64)
    next_foa, duration = gaze._simulate_next_foa(state.curr_foa, prior)
    state.add_fixation(next_foa, duration)


def run_agent(image_path, query, config, llm_client, segmenter, gaze, work_dir,
              image_size=None, verbose=False):
    """Return the scanpath, durations, and stop reason."""
    output_dirs = {"main": str(work_dir), "sam": str(Path(work_dir) / "segments")}
    os.makedirs(output_dirs["sam"], exist_ok=True)
    inference = config["inference"]
    judge_threshold = (inference.get("termination_confidence_threshold", 0.86)
                       if inference.get("judge_enabled", True) else None)
    state = AgentState(config, {}, str(image_path), query, image_size)
    handlers = ToolHandlers(
        state=state, sam_service=segmenter, llm_client=llm_client,
        scanpath_model=gaze, output_dirs=output_dirs,
        prompts=load_prompts(judge_threshold), verbose=verbose,
    )
    messages = create_initial_messages(str(image_path), query, handlers.prompts["system"])
    generated_text = llm_client(messages, agent_role="orchestrator")
    reason = "agent_stopped"

    while generated_text is not None:
        tool_call, generated_text = parse_tool_call_with_retry(
            generated_text=generated_text, messages=messages, llm_client=llm_client,
            agent_role="orchestrator", verbose=verbose,
        )
        if tool_call is None:
            reason = "tool_parse_failed"
            break
        name = tool_call["name"]
        if verbose:
            print(f"Generation {state.generation_count + 1}/{inference['max_generations']}: "
                  f"calling {name} (sampled fixations: {state.fixation_count})...")
        if name == "segment_phrase":
            handlers.handle_segment_phrase(tool_call, messages)
        elif name == "predict_foa":
            handlers.handle_predict_foa(tool_call, messages)
        elif name == "examine_mask":
            handlers.handle_examine_mask(tool_call, messages)
        elif name == "judge":
            if inference.get("judge_enabled", True):
                result = handlers.handle_judge(tool_call, messages)
                if result["confidence"] >= judge_threshold:
                    sample_final_fixation(state, gaze)
                    return state.scanpath[:-1], state.durs, "judge_confident"
            if state.get_total_duration() >= inference["max_duration"]:
                sample_final_fixation(state, gaze)
                return state.scanpath[:-1], state.durs, "threshold_reached"
            else:
                handlers.handle_predict_foa({"name": "predict_foa", "parameters": {}}, messages)
        else:
            raise ValueError(f"Unknown tool call: {name}")

        state.increment_generation_count()
        # These are the original active checks, in their original order.
        if state.generation_count >= inference["max_generations"]:
            reason = "max_generations"
            break
        if state.get_total_duration() > inference["max_duration"]:
            reason = "max_duration"
            break
        messages = prune_messages_for_next_round(
            messages, state.used_text_prompts, state.latest_sam_text_prompt,
            str(image_path), query,
        )
        generated_text = llm_client(messages, agent_role="orchestrator")
        if name == "judge":
            required_tool = {"Fixation_Set": "examine_mask",
                             "Masks_Ready": "predict_foa"}.get(state.curr_state)
            for _ in range(2):
                if (not isinstance(generated_text, str) or
                        not generated_text.strip().lower().startswith("judge completed") or
                        "<tool>" in generated_text):
                    break
                if required_tool is None:
                    break
                messages.extend([
                    {"role": "assistant", "content": [
                        {"type": "text", "text": generated_text}]},
                    {"role": "user", "content": [
                        {"type": "text", "text":
                         "The judge step is complete. The current state is "
                         f"`{state.curr_state}`. Call `{required_tool}` now. Reply "
                         "with exactly one tool call in this format: "
                         f'<tool>{{"name":"{required_tool}","parameters":{{}}}}</tool>'}]},
                ])
                generated_text = llm_client(messages, agent_role="orchestrator")

    return state.scanpath[:-1], state.durs, reason
