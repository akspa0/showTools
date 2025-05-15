import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Determine the project root directory relative to this settings file
# Assumes settings.py is in ClapAnnotator/config/
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Load environment variables from .env file in the project root
dotenv_path = PROJECT_ROOT / '.env'
load_dotenv(dotenv_path=dotenv_path)

# --- Core Settings ---
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    warnings.warn("Hugging Face token (HF_TOKEN) not found in .env file. Access to private models may be restricted.")

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

# --- Model Settings ---
CLAP_MODEL_NAME = "laion/clap-htsat-fused"
CLAP_EXPECTED_SR = 48000 # Expected sample rate for CLAP
CLAP_CHUNK_DURATION_S = 3 # Duration of audio chunks for CLAP processing (in seconds)
DEFAULT_CLAP_CONFIDENCE_THRESHOLD = 0.55 # Default cutoff for CLAP detections

# Define available audio separator models for the UI dropdown
# Structure: 'Display Name': {
#     'model_name': 'internal_name_for_lib',
#     'params': { # Keyword args passed to Separator init
#         # e.g., 'mdx_params': { 'segment_size': 256, 'overlap': 0.25 },
#         # 'vr_params': { 'batch_size': 4, 'aggression': 5 }
#         # 'demucs_params': { 'shifts': 2 }
#         # 'output_format': 'FLAC' # Can override global defaults per model
#     }
# }
# Using defaults from python-audio-separator README for now
AUDIO_SEPARATOR_AVAILABLE_MODELS = {
    "Mel Band RoFormer Vocals": {
        "model_name": "mel_band_roformer_vocals_fv4_gabox.ckpt", 
        "params": {
            "use_soundfile": True
        }
    },
    "UVR-MDX-NET Main": {
        "model_name": "UVR_MDXNET_Main", 
        "params": {
            "mdx_params": {"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 1, "enable_denoise": False}
        }
    },
    "UVR-MDX-NET Inst HQ 1": {
        "model_name": "UVR_MDXNET_Inst_HQ_1", 
        "params": {
            "mdx_params": {"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 1, "enable_denoise": False}
        }
    },
    "Demucs v4 (htdemucs_ft)": {
        "model_name": "htdemucs_ft", 
        "params": {
            "demucs_params": {"segment_size": "Default", "shifts": 2, "overlap": 0.25, "segments_enabled": True}
        }
    },
    # TODO: Add more models and verify/tune default params as needed
    "VR Arch (5_HP-Karaoke-UVR)": { # Example VR Arch model
        "model_name": "5_HP-Karaoke-UVR",
        "params": {
             "vr_params": {"batch_size": 4, "window_size": 512, "aggression": 5, "enable_tta": False, "enable_post_process": False, "post_process_threshold": 0.2, "high_end_process": False}
        }
    }
}
DEFAULT_AUDIO_SEPARATOR_MODEL = "Mel Band RoFormer Vocals" # Must be a key from AUDIO_SEPARATOR_AVAILABLE_MODELS

# --- Path Settings ---
# Use Path objects for better path handling
AUDIO_SEPARATOR_MODEL_DIR = PROJECT_ROOT / '_models' / 'audio-separator' # Local cache for separator models
TEMP_OUTPUT_DIR = PROJECT_ROOT / 'temp_output' # Temporary storage for intermediate files
BASE_OUTPUT_DIR = PROJECT_ROOT / 'ClapAnnotator_Output' # Base directory for final JSON results
CLAP_PRESETS_DIR = PROJECT_ROOT / '_presets' / 'clap_prompts' # Directory for user CLAP presets

# --- Ensure directories exist (optional, can also be done at runtime) ---
# It's often better to ensure directories exist just before they are needed,
# but creating them here can be done for simplicity if preferred.
# TEMP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# CLAP_PRESETS_DIR.mkdir(parents=True, exist_ok=True)
# AUDIO_SEPARATOR_MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"Project Root: {PROJECT_ROOT}")
print(f"Temporary Output Dir: {TEMP_OUTPUT_DIR}")
print(f"Base Output Dir: {BASE_OUTPUT_DIR}")
print(f"CLAP Presets Dir: {CLAP_PRESETS_DIR}")
print(f"Separator Model Dir: {AUDIO_SEPARATOR_MODEL_DIR}") 