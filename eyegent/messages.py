import copy
import json
from typing import List, Dict, Set, Any

def create_initial_messages(img_path: str, initial_text_prompt: str, system_prompt: str) -> List[Dict[str, Any]]:
    return [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': [{'type': 'image', 'image': img_path}, {'type': 'text', 'text': f"User query: '{initial_text_prompt}'"}]}]

def prune_messages_for_next_round(messages_list: List[Dict], used_text_prompts: Set[str], latest_sam3_text_prompt: str, img_path: str, initial_text_prompt: str) -> List[Dict]:
    part1 = copy.deepcopy(messages_list[:2])
    part2_start_idx = None
    for idx in range(len(messages_list) - 1, 1, -1):
        msg = messages_list[idx]
        if msg.get('role') != 'assistant' or 'content' not in msg:
            continue
        for content in msg['content']:
            if isinstance(content, dict) and content.get('type') == 'text' and ('<tool>' in content.get('text', '')) and ('segment_phrase' in content.get('text', '')):
                part2_start_idx = idx
                break
        if part2_start_idx is not None:
            break
    part2 = messages_list[part2_start_idx:] if part2_start_idx is not None else []
    latest_prompts = set((p.strip() for p in latest_sam3_text_prompt.split(','))) if latest_sam3_text_prompt else set()
    previously_used = sorted(used_text_prompts - latest_prompts) if latest_sam3_text_prompt else sorted(used_text_prompts)
    if part2 and len(previously_used) > 0:
        warning_text = f'Note that we have previously called the segment_phrase tool with text_prompts including: {list(previously_used)}, but none of the generated results were satisfactory. Make sure you do not reuse any of these phrases when calling segment_phrase again.'
        part1[1] = {'role': 'user', 'content': [{'type': 'image', 'image': img_path}, {'type': 'text', 'text': f"The above image is the raw input image. The initial user input query is: '{initial_text_prompt}'. {warning_text}"}]}
        assert len(part1[1]['content']) == 2
    new_messages = list(part1)
    new_messages.extend(part2)
    return new_messages

def parse_tool_call_with_retry(generated_text: str, messages: List[Dict], llm_client: Any, agent_role: str='orchestrator', max_retries: int=2, verbose: bool=True) -> tuple:

    def _log(msg: str):
        if verbose:
            print(msg)
    for attempt in range(max_retries + 1):
        if attempt > 0:
            _log(f"\n⚠️ Retry {attempt}/{max_retries} - LLM didn't follow tool format...")
            retry_messages = messages + [{'role': 'assistant', 'content': [{'type': 'text', 'text': generated_text}]}, {'role': 'user', 'content': [{'type': 'text', 'text': 'ERROR: Your response did not follow the required format. You MUST use the exact tool call format:\n\n<tool>\n{"name": "tool_name", "arguments": {...}}\n</tool>\n\nPlease provide your response again in the correct format.'}]}]
            generated_text = llm_client(retry_messages, agent_role=agent_role)
            if generated_text is None:
                _log(f'❌ Retry {attempt} failed: No response from LLM')
                continue
        if '<tool>' not in generated_text:
            _log(f'⚠️ No <tool> tag found (attempt {attempt + 1})')
            if attempt < max_retries:
                _log(f'Preview: {generated_text[:200]}...')
                continue
            else:
                _log(f'❌ All retries exhausted. Full text:\n{generated_text}')
                return (None, None)
        try:
            generated_text_clean = generated_text.split('</tool>', 1)[0] + '</tool>'
            tool_json_str = generated_text_clean.split('<tool>')[-1].split('</tool>')[0].strip().replace('}}}', '}}')
        except Exception as e:
            _log(f'⚠️ Failed to extract tool JSON (attempt {attempt + 1}): {e}')
            if attempt < max_retries:
                continue
            else:
                return (None, None)
        if not tool_json_str:
            _log(f'⚠️ Empty tool JSON (attempt {attempt + 1})')
            if attempt < max_retries:
                _log(f'Preview: {generated_text[:200]}...')
                continue
            else:
                _log(f'❌ All retries exhausted. Full text:\n{generated_text}')
                return (None, None)
        try:
            tool_call = json.loads(tool_json_str)
            if 'name' not in tool_call:
                _log(f"❌ Tool call missing 'name' field: {tool_call}")
                _log('This is a validation error, not a format error. Not retrying.')
                return (None, None)
            if attempt > 0:
                _log('✅ Retry successful! Tool call parsed correctly.')
            tool_call['full_text'] = generated_text_clean
            return (tool_call, generated_text_clean)
        except json.JSONDecodeError as e:
            _log(f'⚠️ Failed to parse JSON (attempt {attempt + 1}): {e}')
            _log(f'Attempted to parse: {tool_json_str[:200]}...')
            if attempt < max_retries:
                continue
            else:
                _log(f'❌ All retries exhausted. Full text:\n{generated_text}')
                return (None, None)
    return (None, None)
