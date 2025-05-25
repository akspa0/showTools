import requests
import json
import hashlib
from pathlib import Path

def run_llm_task(prompt, config, output_path=None, seed=None):
    """
    Run an LLM task using the config dict (same as pipeline_orchestrator.py).
    Returns the LLM response as a string. Optionally writes to output_path.
    """
    base_url = config.get('lm_studio_base_url', 'http://localhost:1234/v1')
    api_key = config.get('lm_studio_api_key', 'lm-studio')
    model_id = config.get('lm_studio_model_identifier', 'llama-3.1-8b-supernova-etherealhermes')
    temperature = config.get('lm_studio_temperature', 0.5)
    max_tokens = config.get('lm_studio_max_tokens', 2048)
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    # Deterministic seed if provided
    if seed is None:
        seed = int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % (2**32)
    data = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed
    }
    try:
        response = requests.post(f"{base_url}/chat/completions", headers=headers, data=json.dumps(data), timeout=60)
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            if output_path:
                Path(output_path).write_text(content, encoding='utf-8')
            return content
        else:
            error_msg = f"LLM API error {response.status_code}: {response.text}"
            if output_path:
                Path(output_path).write_text(error_msg, encoding='utf-8')
            return error_msg
    except Exception as e:
        fail_msg = f"LLM request failed: {e}"
        if output_path:
            Path(output_path).write_text(fail_msg, encoding='utf-8')
        return fail_msg 