import logging
import json
from pathlib import Path
from typing import List, Dict, Union
from config import settings
from .file_utils import ensure_dir, sanitize_filename

log = logging.getLogger(__name__)

PRESET_FILE_EXTENSION = ".txt" # Using .txt for simple line-by-line prompts
# PRESET_FILE_EXTENSION = ".json" # Could also use .json for more complex presets

def load_clap_prompt_presets() -> Dict[str, List[str]]:
    """Loads all CLAP prompt presets from the CLAP_PRESETS_DIR.

    Presets are expected to be text files (.txt), where each line is a prompt.
    The filename (without extension) is used as the preset name.

    Returns:
        A dictionary where keys are preset names and values are lists of prompts.
    """
    presets_dir = settings.CLAP_PRESETS_DIR
    ensure_dir(presets_dir) # Ensure the directory exists
    
    loaded_presets: Dict[str, List[str]] = {}
    log.info(f"Loading CLAP prompt presets from: {presets_dir}")
    
    for preset_file in presets_dir.glob(f"*{PRESET_FILE_EXTENSION}"):
        if preset_file.is_file():
            preset_name = preset_file.stem # Filename without extension
            try:
                with open(preset_file, 'r', encoding='utf-8') as f:
                    prompts = [line.strip() for line in f if line.strip()] # Read non-empty lines
                if prompts:
                    loaded_presets[preset_name] = prompts
                    log.debug(f"Loaded preset '{preset_name}' with {len(prompts)} prompts from {preset_file.name}")
                else:
                    log.warning(f"Preset file '{preset_file.name}' is empty or contains only whitespace. Skipping.")        
            except Exception as e:
                log.error(f"Failed to load or parse preset file {preset_file.name}: {e}")
    
    if not loaded_presets:
        log.info("No CLAP prompt presets found or loaded.")
    return loaded_presets

def save_clap_prompt_preset(preset_name: str, prompts: List[str]) -> bool:
    """Saves a list of CLAP prompts as a preset file.

    The preset name will be sanitized for use as a filename.
    The file will be saved in the CLAP_PRESETS_DIR with a .txt extension.
    Existing presets with the same sanitized name will be overwritten.

    Args:
        preset_name: The desired name for the preset.
        prompts: A list of prompt strings.

    Returns:
        True if saving was successful, False otherwise.
    """
    if not preset_name.strip():
        log.error("Preset name cannot be empty.")
        return False
    if not prompts:
        log.warning("Attempted to save an empty list of prompts. Aborting save.")
        return False

    presets_dir = settings.CLAP_PRESETS_DIR
    ensure_dir(presets_dir)
    
    sanitized_name = sanitize_filename(preset_name) # Sanitize for filename usage
    if not sanitized_name or sanitized_name == "invalid_filename":
        log.error(f"Invalid preset name after sanitization: '{preset_name}' -> '{sanitized_name}'")
        return False
        
    preset_file_path = presets_dir / f"{sanitized_name}{PRESET_FILE_EXTENSION}"
    
    log.info(f"Saving CLAP prompt preset '{preset_name}' (as '{preset_file_path.name}') with {len(prompts)} prompts.")
    try:
        with open(preset_file_path, 'w', encoding='utf-8') as f:
            for prompt in prompts:
                f.write(f"{prompt.strip()}\n") # Write each prompt on a new line
        log.info(f"Successfully saved preset to {preset_file_path}")
        return True
    except Exception as e:
        log.error(f"Failed to save preset file {preset_file_path}: {e}")
        return False

# Example Usage (for testing purposes):
if __name__ == '__main__':
    # This block will only run if you execute this script directly (e.g., python utils/preset_utils.py)
    # Make sure to call setup_logging() if you want to see log messages here.
    # from .logging_config import setup_logging
    # setup_logging() 

    log.info("Testing preset utilities...")
    
    # Test saving
    test_prompts1 = ["sound of a dog barking", "cat meowing loudly", "  silence  "]
    save_clap_prompt_preset("Animal Sounds!?.txt", test_prompts1)
    
    test_prompts2 = ["engine running", "car horn", "tires screeching"]
    save_clap_prompt_preset("Vehicle Noises", test_prompts2)

    save_clap_prompt_preset("Empty Preset", []) # Should warn and not save
    save_clap_prompt_preset("  ", ["a prompt"]) # Should fail sanitization or be 'invalid_filename'

    # Test loading
    loaded = load_clap_prompt_presets()
    if loaded:
        log.info("\nLoaded presets:")
        for name, prpts in loaded.items():
            log.info(f"  Preset: {name}")
            for p in prpts:
                log.info(f"    - {p}")
    else:
        log.info("No presets were loaded.") 