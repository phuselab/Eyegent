# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# Original payload, error handling, and generation parameters; explicit configuration.

import base64
import os
from openai import OpenAI

def get_image_base64_and_mime(image_path):
    try:
        ext = os.path.splitext(image_path)[1].lower()
        mime_types = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.gif': 'image/gif', '.webp': 'image/webp', '.bmp': 'image/bmp'}
        mime_type = mime_types.get(ext, 'image/jpeg')
        with open(image_path, 'rb') as image_file:
            base64_data = base64.b64encode(image_file.read()).decode('utf-8')
            return (base64_data, mime_type)
    except Exception as e:
        print(f'Error converting image to base64: {e}')
        return (None, None)

def _send_generate_request(messages, config, server_url: str | None=None, model: str | None=None, api_key: str | None=None, max_tokens: int | None=None, agent_role: str | None=None, temperature: float | None=None, verbose: bool=False):
    processed_messages = []
    for message in messages:
        processed_message = message.copy()
        if message['role'] == 'user' and 'content' in message:
            processed_content = []
            for c in message['content']:
                if isinstance(c, dict) and c.get('type') == 'image':
                    image_path = c['image']
                    if verbose:
                        print('image_path', image_path)
                    new_image_path = image_path.replace('?', '%3F')
                    try:
                        base64_image, mime_type = get_image_base64_and_mime(new_image_path)
                        if base64_image is None:
                            print(f'Warning: Could not convert image to base64: {new_image_path}')
                            continue
                        processed_content.append({'type': 'image_url', 'image_url': {'url': f'data:{mime_type};base64,{base64_image}', 'detail': 'high'}})
                    except FileNotFoundError:
                        print(f'Warning: Image file not found: {new_image_path}')
                        continue
                    except Exception as e:
                        print(f'Warning: Error processing image {new_image_path}: {e}')
                        continue
                else:
                    processed_content.append(c)
            processed_message['content'] = processed_content
        processed_messages.append(processed_message)
    eff = config
    if model is None:
        model = eff.get('model')
    if api_key is None:
        api_key = eff.get('api_key')
    if server_url is None:
        server_url = eff.get('server_url')
    if max_tokens is None:
        max_tokens = {'llm': config}.get('llm', {}).get('max_tokens', 4096)
    if temperature is None:
        temperature = {'llm': config}.get('llm', {}).get('temperature', 1)
    cfg = {'llm': config}
    thinking_config = cfg.get('llm', {}).get('thinking', {})
    thinking_enabled = thinking_config.get('enabled', False)
    budgets = thinking_config.get('budgets', {})
    if agent_role and agent_role in budgets:
        thinking_budget = budgets[agent_role]
    else:
        thinking_budget = budgets.get('default', None)
    client = OpenAI(api_key=api_key, base_url=server_url)
    try:
        agent_label = f' [{agent_role}]' if agent_role else ''
        if verbose:
            print(f'\n🔍 Calling model {model}{agent_label}...')
        api_params = {'model': model, 'messages': processed_messages, 'max_completion_tokens': max_tokens, 'n': 1, 'temperature': temperature}
        if eff.get('seed') is not None:
            api_params['seed'] = eff['seed']
        if thinking_enabled:
            extra_body = {'enable_thinking': True}
            if thinking_budget is not None:
                extra_body['thinking_budget'] = thinking_budget
            api_params['extra_body'] = extra_body
            if verbose:
                budget_str = str(thinking_budget) if thinking_budget else 'unlimited'
                print(f'💭 Thinking enabled with budget: {budget_str} tokens')
        response = client.chat.completions.create(**api_params)
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            print(f'Unexpected response format: {response}')
            return None
    except Exception as e:
        print(f'Request failed: {e}')
        return None

class VLMClient:

    def __init__(self, config):
        self.config = config

    def __call__(self, messages, agent_role=None):
        return _send_generate_request(messages, self.config, agent_role=agent_role)
