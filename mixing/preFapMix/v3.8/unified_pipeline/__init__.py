import os
import yaml
from pathlib import Path

# CLAP prompt presets
CLAP_PRESET_DIR = Path(__file__).resolve().parent.parent / 'ClapAnnotator' / '_presets' / 'clap_prompts'
# Full pipeline presets
PIPELINE_PRESET_DIR = Path(__file__).resolve().parent / 'presets'
PIPELINE_PRESET_DIR.mkdir(exist_ok=True)

def list_clap_presets():
    return [f.stem for f in CLAP_PRESET_DIR.glob('*.txt') if f.is_file()]

def load_clap_preset(name):
    preset_path = CLAP_PRESET_DIR / f'{name}.txt'
    if not preset_path.exists():
        raise FileNotFoundError(f"CLAP preset '{name}' not found.")
    with open(preset_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

def save_clap_preset(name, prompts):
    preset_path = CLAP_PRESET_DIR / f'{name}.txt'
    with open(preset_path, 'w', encoding='utf-8') as f:
        for prompt in prompts:
            f.write(f"{prompt}\n")

def list_pipeline_presets():
    return [f.stem for f in PIPELINE_PRESET_DIR.glob('*.yaml') if f.is_file()]

def load_pipeline_preset(name):
    preset_path = PIPELINE_PRESET_DIR / f'{name}.yaml'
    if not preset_path.exists():
        raise FileNotFoundError(f"Pipeline preset '{name}' not found.")
    with open(preset_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_pipeline_preset(name, config_dict):
    preset_path = PIPELINE_PRESET_DIR / f'{name}.yaml'
    with open(preset_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, allow_unicode=True, sort_keys=False) 