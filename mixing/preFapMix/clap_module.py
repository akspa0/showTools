import logging
import json
from pathlib import Path
import sys # Added for sys.path manipulation
import os # Added for path joining
import shutil # Added for file copying

# Placeholder for actual CLAP annotation logic
# Expected to take an audio file, process it, and save results (e.g., a JSON)
# in the output_dir_str.
# It should return a dictionary, and the workflow executor will look for a key
# (defined in the workflow JSON's "outputs" section) in this dictionary
# to get the path to the primary output file(s).

log = logging.getLogger(__name__)

def run_clap_annotation(input_audio_path: str, output_dir_str: str, pii_safe_file_prefix: str, config: dict) -> dict:
    log.info(f"[CLAP Module] Attempting to run CLAP annotation for: {input_audio_path}")
    log.info(f"[CLAP Module] PII-Safe Prefix: {pii_safe_file_prefix}")
    log.info(f"[CLAP Module] Config: {config}")
    log.info(f"[CLAP Module] Output directory: {output_dir_str}")

    clap_annotator_base_path = Path(__file__).parent.resolve() / "v3.8" / "ClapAnnotator"
    temp_pii_safe_input_audio_path = None # For cleanup

    # Temporarily add ClapAnnotator path to sys.path
    sys_path_modified = False
    if str(clap_annotator_base_path) not in sys.path:
        sys.path.insert(0, str(clap_annotator_base_path))
        sys_path_modified = True
        log.debug(f"Added {clap_annotator_base_path} to sys.path")

    try:
        from cli import process_audio_file
        # Import settings to access defaults if needed, or rely on cli.py to use them
        from config import settings as clap_settings

        input_file_path = Path(input_audio_path)
        output_path = Path(output_dir_str)
        output_path.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

        # Create a temporary, PII-safe copy of the input audio to pass to ClapAnnotator
        # This ensures the output JSON from ClapAnnotator also uses a PII-safe stem.
        temp_parent_dir = output_path # Place temp audio in the stage's output dir for simplicity in this case
        temp_pii_safe_input_audio_path = temp_parent_dir / f"{pii_safe_file_prefix}_temp_clap_input{input_file_path.suffix}"
        
        try:
            shutil.copy2(input_file_path, temp_pii_safe_input_audio_path)
            log.info(f"[CLAP Module] Created temporary PII-safe input audio for ClapAnnotator: {temp_pii_safe_input_audio_path}")
        except Exception as e_copy:
            log.error(f"[CLAP Module] Failed to create temporary PII-safe input copy: {e_copy}")
            return None

        # Get prompts and confidence from config, with defaults
        prompts = config.get('clap_prompts', ["speech", "music", "sound effect", "vocals", "instrumental"])
        if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            log.warning(f"Invalid 'clap_prompts' in config: {prompts}. Using defaults.")
            prompts = ["speech", "music", "sound effect", "vocals", "instrumental"]

        confidence_threshold = config.get('clap_confidence_threshold', clap_settings.DEFAULT_CLAP_CONFIDENCE_THRESHOLD)
        try:
            confidence_threshold = float(confidence_threshold)
        except ValueError:
            log.warning(f"Invalid 'clap_confidence_threshold': {confidence_threshold}. Using default: {clap_settings.DEFAULT_CLAP_CONFIDENCE_THRESHOLD}")
            confidence_threshold = clap_settings.DEFAULT_CLAP_CONFIDENCE_THRESHOLD
        
        separator_model_name = config.get('clap_separator_model', clap_settings.DEFAULT_AUDIO_SEPARATOR_MODEL)
        if separator_model_name not in clap_settings.AUDIO_SEPARATOR_AVAILABLE_MODELS:
            log.warning(f"Separator model '{separator_model_name}' not found in ClapAnnotator settings. Using default: {clap_settings.DEFAULT_AUDIO_SEPARATOR_MODEL}")
            separator_model_name = clap_settings.DEFAULT_AUDIO_SEPARATOR_MODEL

        log.info(f"[CLAP Module] Using prompts: {prompts}")
        log.info(f"[CLAP Module] Using confidence threshold: {confidence_threshold}")
        log.info(f"[CLAP Module] Using separator model: {separator_model_name}")
        log.info(f"[CLAP Module] Target output dir for ClapAnnotator: {str(output_path)}")


        # Call the process_audio_file function from ClapAnnotator
        # The output_dir here IS the final directory for the _clap_results.json
        # It does not create an additional subfolder if output_dir is specified.
        result_data = process_audio_file(
            input_file=temp_pii_safe_input_audio_path, # Use the PII-safe temp copy
            separator_model=separator_model_name, # Use a default or make configurable
            prompts=prompts,
            confidence_threshold=confidence_threshold,
            output_dir=output_path, # Pass the stage's output dir directly
            cleanup_temp=True,      # Clean up ClapAnnotator's temp files
            keep_audio=False,       # We don't need copies of audio in the stage output
            chunk_duration=clap_settings.CLAP_CHUNK_DURATION_S
        )

        if result_data.get("success"):
            output_json_path = result_data.get("output_file")
            log.info(f"[CLAP Module] ClapAnnotator processing successful. Results JSON: {output_json_path}")
            # The key "clap_events_file" must match what's in the workflow JSON "outputs"
            return {"clap_events_file": str(output_json_path)}
        else:
            error_msg = result_data.get("error", "Unknown error during ClapAnnotator processing.")
            log.error(f"[CLAP Module] ClapAnnotator processing failed: {error_msg}")
            log.error(f"[CLAP Module] Traceback: {result_data.get('traceback')}")
            # Propagate failure: return None or a dict indicating error
            # The workflow executor checks if the function returned None or if a required output key is missing.
            return None # Indicates failure

    except ImportError as e:
        log.error(f"[CLAP Module] Failed to import from ClapAnnotator: {e}")
        log.error("[CLAP Module] Ensure ClapAnnotator is at v3.8/ClapAnnotator relative to project root and sys.path was updated correctly.")
        return None
    except Exception as e:
        log.exception(f"[CLAP Module] An unexpected error occurred: {e}")
        return None
    finally:
        # Clean up sys.path modification
        if sys_path_modified and str(clap_annotator_base_path) in sys.path:
            sys.path.pop(0)
            log.debug(f"Removed {clap_annotator_base_path} from sys.path")
        
        # Clean up the temporary PII-safe input audio file
        if temp_pii_safe_input_audio_path and temp_pii_safe_input_audio_path.exists():
            try:
                os.remove(temp_pii_safe_input_audio_path)
                log.info(f"[CLAP Module] Cleaned up temporary PII-safe input audio: {temp_pii_safe_input_audio_path}")
            except OSError as e_remove:
                log.warning(f"[CLAP Module] Failed to remove temporary PII-safe input audio {temp_pii_safe_input_audio_path}: {e_remove}")

# For basic testing if run directly
if __name__ == '__main__':
    # Configure basic logging for direct script run
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create a dummy pipeline_test_input.wav if it doesn't exist for testing
    dummy_audio_path = Path(__file__).parent / "pipeline_test_input.wav"
    if not dummy_audio_path.exists():
        try:
            import soundfile as sf
            import numpy as np
            samplerate = 44100
            duration = 1
            frequency = 440
            t = np.linspace(0, duration, int(samplerate * duration), False)
            data = 0.5 * np.sin(2 * np.pi * frequency * t)
            sf.write(dummy_audio_path, data, samplerate)
            log.info(f"Created dummy audio file: {dummy_audio_path}")
        except Exception as e:
            log.error(f"Could not create dummy audio file: {e}")

    if dummy_audio_path.exists():
        # Define a test config
        test_output_dir = Path(__file__).parent / "test_clap_output"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        test_config = {
            "clap_prompts": ["siren", "alarm", "speech"],
            "clap_confidence_threshold": 0.6,
            "clap_separator_model": "UVR-MDX-NET Main" # Example, ensure it's valid in clap_settings
        }
        
        log.info("--- Running CLAP Module Test ---")
        results = run_clap_annotation(
            input_audio_path=str(dummy_audio_path),
            output_dir_str=str(test_output_dir),
            pii_safe_file_prefix="test_clap_prefix", # Add dummy prefix for testing
            config=test_config
        )
        log.info("--- CLAP Module Test Finished ---")
        
        if results and results.get("clap_events_file"):
            log.info(f"Test successful. Output JSON: {results['clap_events_file']}")
            # Optionally, print content of JSON
            # with open(results['clap_events_file'], 'r') as f:
            #     print(json.load(f))
        else:
            log.error("Test failed or no output file produced.")
    else:
        log.error(f"Test input audio {dummy_audio_path} not found. Cannot run test.") 