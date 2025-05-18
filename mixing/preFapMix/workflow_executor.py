import os
import json
import logging
import importlib
from pathlib import Path
from datetime import datetime
import shutil
import re
import argparse
import subprocess
import threading
import queue
import soundfile as sf
import numpy as np
import mutagen
from final_output_builder import build_final_output
import yt_dlp

# Global logger for the executor
logger = logging.getLogger(__name__)

# --- For Debugging: Set to a stage name to run only that stage, or None to run all --- 
DEBUG_SINGLE_STAGE_NAME = None # Changed from clap_event_annotation
# Example stage names: "clap_event_annotation", "audio_preprocessing", "speaker_diarization", "transcription", "llm_summary_and_analysis"

def setup_workflow_logging(log_level_str='INFO', log_file=None):
    """Configures logging for the workflow executor."""
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logging.basicConfig(level=log_level, 
                        format='%(asctime)s [%(levelname)s] - %(name)s.%(funcName)s - %(message)s',
                        handlers=[logging.StreamHandler()]) # Add console handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] - %(name)s.%(funcName)s - %(message)s'))
        logging.getLogger().addHandler(file_handler) # Add file handler to root logger
    # Suppress overly verbose logs from libraries if needed
    # logging.getLogger("some_library").setLevel(logging.WARNING)

def _generate_pii_safe_run_identifier_stem(filename: str) -> str:
    """
    Generates a PII-safe identifier stem from a filename, prioritizing timestamps.
    Adapted from audiotoolkit_phase1.py's generate_call_id logic.
    """
    # Try to extract a timestamp like YYYYMMDD-HHMMSS
    match_timestamp = re.search(r'(\d{8}-\d{6})-\d{10}(?:\.\d+)?', filename)
    if match_timestamp:
        return f"call_{match_timestamp.group(1)}" # e.g., call_20250512-020958
    else:
        # Fallback to sanitized filename part if no specific timestamp pattern is found.
        # This might still leak some PII if the original filename didn't have the expected timestamp pattern.
        # Consider if a more generic timestamp extraction or just a generic ID is better for PII safety here.
        # For now, using a sanitized version of the original stem.
        sanitized_stem = re.sub(r'[^a-zA-Z0-9_-]', '', Path(filename).stem)
        # Add a generic prefix to indicate it's a processed ID.
        # If even the sanitized_stem is too revealing, a completely opaque ID would be needed.
        # For this adaptation, let's keep it somewhat traceable if the timestamp isn't there.
        logger.warning(f"Could not extract YYYYMMDD-HHMMSS timestamp from '{filename}' for PII-safe ID. Using sanitized stem: '{sanitized_stem}'.")
        return f"item_{sanitized_stem[:50]}"

def _resolve_input_path(value, workflow_context, current_stage_output_dir, original_input_audio_file, main_run_dir):
    """Resolves special string tokens to actual paths or values."""
    if not isinstance(value, str):
        return value # Not a string, return as is

    if value == "{workflow.original_input_audio_file}":
        return str(original_input_audio_file)
    if value == "{workflow.main_run_dir}":
        return str(main_run_dir)
    if value == "{workflow.current_stage_output_dir}":
        return str(current_stage_output_dir)
    
    if value.startswith("{stages.") and value.endswith("}"):
        path_expression = value[8:-1] # e.g., "audio_preprocessing.processed_stems_info[vocals_normalized_path]"
        
        # Extract stage_name_ref (part before the first . or [)
        match_stage_name = re.match(r"(\w+)", path_expression)
        if not match_stage_name:
            logger.error(f"Could not extract stage name from path expression: {path_expression}")
            return None
        stage_name_ref = match_stage_name.group(1)
        
        if stage_name_ref not in workflow_context:
            logger.error(f"Referenced stage '{stage_name_ref}' not found in workflow context. Available: {list(workflow_context.keys())}")
            return None
        
        current_val = workflow_context[stage_name_ref]
        
        # The rest of the path expression after the stage name
        remaining_expression = path_expression[len(stage_name_ref):]
        
        # Tokenize the remaining expression: sequence of .key or [key_or_index]
        # Regex: finds either .followed_by_word or [anything_not_closing_bracket]
        accessor_tokens = re.finditer(r'\.(\w+)|\[([^\]]+)\]', remaining_expression)
        
        for token_match in accessor_tokens:
            if current_val is None:
                logger.warning(f"Cannot resolve further parts of '{value}'; a preceding part resolved to None.")
                return None

            dot_key = token_match.group(1)       # Matched .key
            bracket_key_str = token_match.group(2) # Matched [key_or_index]
            
            if dot_key: # Access with dot key
                if not isinstance(current_val, dict) or dot_key not in current_val:
                    logger.error(f"Key '{dot_key}' not found or not a dict during traversal of '{value}'. Current value type: {type(current_val)}")
                    return None
                current_val = current_val[dot_key]
            elif bracket_key_str is not None: # Access with bracket key/index (can be string or needs int conversion for list)
                try:
                    # Attempt to convert to int for list index first
                    if isinstance(current_val, list):
                        try:
                            idx = int(bracket_key_str)
                            current_val = current_val[idx]
                        except ValueError:
                            logger.error(f"Invalid list index '{bracket_key_str}' (not an integer) in '{value}'.")
                            return None
                        except IndexError:
                            logger.error(f"List index '{bracket_key_str}' out of range in '{value}'.")
                            return None
                    elif isinstance(current_val, dict):
                        logger.debug(f"Checking for key: '{bracket_key_str}' (type: {type(bracket_key_str)}) in dict keys: {list(current_val.keys())}")
                        if bracket_key_str not in current_val:
                             logger.error(f"Key '{bracket_key_str}' not found in dict during traversal of '{value}'. Dict was: {current_val}")
                             return None
                        current_val = current_val[bracket_key_str]
                    else:
                        logger.error(f"Cannot apply index/key '{bracket_key_str}' to non-subscriptable type {type(current_val)} in '{value}'.")
                        return None
                except (KeyError, IndexError) as e: # Should be caught by specific checks above, but as a safeguard
                    logger.error(f"Error accessing '{bracket_key_str}' in path '{value}': {e}")
                    return None
            # If neither dot_key nor bracket_key_str matched, it's an issue with the regex or path, but finditer shouldn't yield such matches.

        return current_val
    
    return value # Not a special string handled by stages, return as is

def _resolve_stage_inputs(input_defs, workflow_context, current_stage_output_dir, original_input_audio_file, main_run_dir):
    """Resolves all inputs for a stage."""
    resolved_inputs = {}
    if input_defs:
        for key, val_template in input_defs.items():
            resolved_inputs[key] = _resolve_input_path(val_template, workflow_context, current_stage_output_dir, original_input_audio_file, main_run_dir)
            if resolved_inputs[key] is None and isinstance(val_template, str) and val_template.startswith("{"):
                logger.warning(f"Input '{key}' resolved to None from template '{val_template}'. This might cause issues.")
    return resolved_inputs

def execute_workflow(workflow_file_path: str, input_audio_file_str: str, base_output_dir_str: str, global_log_level='INFO'):
    """
    Parses and executes an audio processing workflow defined in a JSON file.

    Args:
        workflow_file_path: Path to the workflow JSON file.
        input_audio_file_str: Path to the input audio file.
        base_output_dir_str: Base directory where run-specific output folders will be created.
        global_log_level: Logging level for the executor and potentially modules.
    """
    input_audio_file = Path(input_audio_file_str)
    base_output_dir = Path(base_output_dir_str)

    if not input_audio_file.is_file():
        logger.error(f"Input audio file not found: {input_audio_file}")
        return False

    try:
        with open(workflow_file_path, 'r') as f:
            workflow_config = json.load(f)
    except FileNotFoundError:
        logger.error(f"Workflow file not found: {workflow_file_path}")
        return False
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding workflow JSON from {workflow_file_path}: {e}")
        return False

    workflow_name = workflow_config.get("name", Path(workflow_file_path).stem)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate PII-safe identifier for the run directory
    pii_safe_identifier_stem = _generate_pii_safe_run_identifier_stem(input_audio_file.name)
    run_id = f"{workflow_name}_{pii_safe_identifier_stem}_{run_timestamp}"
    
    main_run_dir = base_output_dir / run_id
    
    try:
        main_run_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create main run directory {main_run_dir}: {e}")
        return False

    # Setup logging for this run, including a file in the run directory
    run_log_file = main_run_dir / f"{run_id}_executor.log"
    setup_workflow_logging(global_log_level, run_log_file) # Configures root logger
    
    logger.info(f"Starting workflow '{workflow_name}' for input: {input_audio_file}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Main output directory: {main_run_dir}")
    logger.info(f"Logging to console and to: {run_log_file}")

    workflow_context = {"workflow_start_time": run_timestamp} 
    failed_stages_info = {} # Initialize dictionary to store info about failed stages
    stages = workflow_config.get("stages", [])

    for i, stage_def in enumerate(stages):
        stage_name = stage_def.get("stage_name", f"stage_{i:02d}")
        module_name = stage_def.get("module")
        function_name = stage_def.get("function")
        stage_failed = False # Initialize stage_failed for each stage
        
        # --- Debugging: Conditional stage execution ---
        if DEBUG_SINGLE_STAGE_NAME and stage_name != DEBUG_SINGLE_STAGE_NAME:
            logger.info(f"--- SKIPPING Stage {i+1}/{len(stages)}: {stage_name} (DEBUG_SINGLE_STAGE_NAME is set to '{DEBUG_SINGLE_STAGE_NAME}') ---")
            workflow_context[stage_name] = {"status": "skipped_due_to_debug_flag", "output": None} # Add to context so it exists
            continue
        # --- End Debugging ---

        logger.info(f"--- Starting Stage {i+1}/{len(stages)}: {stage_name} ---")
        
        if not module_name or not function_name:
            logger.error(f"Stage '{stage_name}' is missing 'module' or 'function'. Skipping.")
            workflow_context[stage_name] = {"error": "Missing module or function definition"}
            failed_stages_info[stage_name] = "Missing module/function"
            continue

        current_stage_output_dir = main_run_dir / f"{i:02d}_{stage_name}"
        try:
            current_stage_output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create stage output directory {current_stage_output_dir}: {e}. Skipping stage.")
            workflow_context[stage_name] = {"error": f"Failed to create stage output directory: {e}"}
            failed_stages_info[stage_name] = "Failed to create output dir"
            continue
        
        logger.debug(f"Stage output directory: {current_stage_output_dir}")

        try:
            # Resolve inputs for the current stage function
            stage_inputs_config = stage_def.get("inputs", {})
            resolved_inputs = _resolve_stage_inputs(stage_inputs_config, workflow_context, current_stage_output_dir, input_audio_file, main_run_dir)

            function_specific_config = stage_def.get("config", {})
            args_for_function = {**resolved_inputs}
            args_for_function['pii_safe_file_prefix'] = pii_safe_identifier_stem 
            if function_specific_config:
                 args_for_function['config'] = function_specific_config

            logger.debug(f"Importing module: {module_name}, function: {function_name}")
            module_to_import = importlib.import_module(module_name)
            function_to_call = getattr(module_to_import, function_name)
            
            logger.info(f"Executing: {module_name}.{function_name}")
            logger.debug(f"With arguments (first level): { {k: type(v) for k, v in args_for_function.items()} }")

            # Special handling for CLAP segmentation output
            if stage_name == "clap_event_annotation":
                stage_result = function_to_call(**args_for_function)
                # Check for new segmentation output
                if stage_result and "clap_segments_summary" in stage_result:
                    summary_path = Path(stage_result["clap_segments_summary"])
                    with open(summary_path, 'r') as sf:
                        segments = json.load(sf)
                    logger.info(f"CLAP segmentation produced {len(segments)} segments. Launching virtual calls for each segment.")
                    # For each segment, run the rest of the pipeline as a virtual call
                    downstream_stages = stages[i+1:]
                    for seg_idx, seg in enumerate(segments):
                        seg_audio = seg["segment_wav"]
                        seg_meta = seg["meta_json"]
                        seg_prefix = Path(seg_audio).stem
                        logger.info(f"Processing segment {seg_idx+1}/{len(segments)}: {seg_audio}")
                        seg_run_dir = main_run_dir / f"segment_{seg_idx+1:03d}_{seg_prefix}"
                        seg_run_dir.mkdir(parents=True, exist_ok=True)
                        seg_context = {"segment_audio": seg_audio, "segment_meta": seg_meta}
                        # For each downstream stage, run as usual but override input_audio_file to segment
                        seg_workflow_context = dict(workflow_context) # Copy context up to this point
                        seg_workflow_context["clap_event_annotation"] = {"segment_audio": seg_audio, "segment_meta": seg_meta}
                        seg_input_audio_file = Path(seg_audio)
                        for j, downstream_stage_def in enumerate(downstream_stages):
                            ds_stage_name = downstream_stage_def.get("stage_name", f"stage_{i+1+j:02d}")
                            ds_module_name = downstream_stage_def.get("module")
                            ds_function_name = downstream_stage_def.get("function")
                            ds_stage_output_dir = seg_run_dir / f"{j:02d}_{ds_stage_name}"
                            ds_stage_output_dir.mkdir(parents=True, exist_ok=True)
                            ds_stage_inputs_config = downstream_stage_def.get("inputs", {})
                            # Override input_audio_file for this segment
                            ds_resolved_inputs = _resolve_stage_inputs(ds_stage_inputs_config, seg_workflow_context, ds_stage_output_dir, seg_input_audio_file, seg_run_dir)
                            ds_args_for_function = {**ds_resolved_inputs}
                            ds_args_for_function['pii_safe_file_prefix'] = seg_prefix
                            ds_function_specific_config = downstream_stage_def.get("config", {})
                            if ds_function_specific_config:
                                ds_args_for_function['config'] = ds_function_specific_config
                            ds_module_to_import = importlib.import_module(ds_module_name)
                            ds_function_to_call = getattr(ds_module_to_import, ds_function_name)
                            logger.info(f"[Segment {seg_idx+1}] Executing: {ds_module_name}.{ds_function_name}")
                            ds_stage_result = ds_function_to_call(**ds_args_for_function)
                            # Store results in seg_workflow_context for downstream input resolution
                            if downstream_stage_def.get("outputs"):
                                for context_key, result_key_expression in downstream_stage_def["outputs"].items():
                                    if result_key_expression == "return_value":
                                        if ds_stage_name not in seg_workflow_context or not isinstance(seg_workflow_context.get(ds_stage_name), dict):
                                            seg_workflow_context[ds_stage_name] = {}
                                        seg_workflow_context[ds_stage_name][context_key] = ds_stage_result
                                    elif isinstance(ds_stage_result, dict) and result_key_expression in ds_stage_result:
                                        if ds_stage_name not in seg_workflow_context or not isinstance(seg_workflow_context.get(ds_stage_name), dict):
                                            seg_workflow_context[ds_stage_name] = {}
                                        seg_workflow_context[ds_stage_name][context_key] = ds_stage_result[result_key_expression]
                                    elif isinstance(ds_stage_result, dict) and result_key_expression.startswith("return_value[") and result_key_expression.endswith("]"):
                                        actual_key = result_key_expression[len("return_value["):-1]
                                        if actual_key in ds_stage_result:
                                            if ds_stage_name not in seg_workflow_context or not isinstance(seg_workflow_context.get(ds_stage_name), dict):
                                                seg_workflow_context[ds_stage_name] = {}
                                            seg_workflow_context[ds_stage_name][context_key] = ds_stage_result[actual_key]
                    logger.info(f"All segments processed. Skipping downstream stages for original file.")
                    break # Do not process downstream stages for the original file
                else:
                    # Fallback: old single-file flow
                    if stage_def.get("outputs"):
                        for context_key, result_key_expression in stage_def["outputs"].items():
                            if result_key_expression == "return_value":
                                if stage_name not in workflow_context or not isinstance(workflow_context.get(stage_name), dict):
                                    workflow_context[stage_name] = {}
                                workflow_context[stage_name][context_key] = stage_result
                            elif isinstance(stage_result, dict) and result_key_expression in stage_result:
                                if stage_name not in workflow_context or not isinstance(workflow_context.get(stage_name), dict):
                                    workflow_context[stage_name] = {}
                                workflow_context[stage_name][context_key] = stage_result[result_key_expression]
                            elif isinstance(stage_result, dict) and result_key_expression.startswith("return_value[") and result_key_expression.endswith("]"):
                                actual_key = result_key_expression[len("return_value["):-1]
                                if actual_key in stage_result:
                                    if stage_name not in workflow_context or not isinstance(workflow_context.get(stage_name), dict):
                                        workflow_context[stage_name] = {}
                                    workflow_context[stage_name][context_key] = stage_result[actual_key]
            else:
                # Normal stage execution for non-CLAP stages
                stage_result = function_to_call(**args_for_function)
                if stage_def.get("outputs"):
                    for context_key, result_key_expression in stage_def["outputs"].items():
                        if result_key_expression == "return_value":
                            if stage_name not in workflow_context or not isinstance(workflow_context.get(stage_name), dict):
                                workflow_context[stage_name] = {}
                            workflow_context[stage_name][context_key] = stage_result
                        elif isinstance(stage_result, dict) and result_key_expression in stage_result:
                            if stage_name not in workflow_context or not isinstance(workflow_context.get(stage_name), dict):
                                workflow_context[stage_name] = {}
                            workflow_context[stage_name][context_key] = stage_result[result_key_expression]
                        elif isinstance(stage_result, dict) and result_key_expression.startswith("return_value[") and result_key_expression.endswith("]"):
                            actual_key = result_key_expression[len("return_value["):-1]
                            if actual_key in stage_result:
                                if stage_name not in workflow_context or not isinstance(workflow_context.get(stage_name), dict):
                                    workflow_context[stage_name] = {}
                                workflow_context[stage_name][context_key] = stage_result[actual_key]
            
            if stage_failed: # Check if any output mapping failed
                logger.error(f"--- Stage {stage_name} failed due to missing output mapping. Context set to None. ---")
                workflow_context[stage_name] = None # Explicitly set context to None for this stage
            elif not stage_failed and stage_result is not None: # Only log success if no mapping errors AND function didn't return None
                logger.info(f"--- Stage {stage_name} completed successfully ---")
            elif stage_result is None: # Explicitly check for None return from function after output mapping attempt
                logger.error(f"--- Stage {stage_name} (function {module_name}.{function_name}) returned None, indicating failure. Context set to None. ---")
                workflow_context[stage_name] = None # Explicitly set context to None
                stage_failed = True # Ensure stage_failed is true if function returned None

        except ImportError as e:
            logger.error(f"Failed to import module {module_name} for stage {stage_name}: {e}")
            workflow_context[stage_name] = {"error": f"ImportError: {e}"}
        except AttributeError as e:
            logger.error(f"Failed to find function {function_name} in module {module_name} for stage {stage_name}: {e}")
            workflow_context[stage_name] = {"error": f"AttributeError: {e}"}
        except Exception as e:
            logger.error(f"Error executing stage {stage_name} ({module_name}.{function_name}): {e}")
            logger.error("Exception details for stage failure:", exc_info=True)
            workflow_context[stage_name] = None # Ensure context is None on any exception
            stage_failed = True # Mark stage as failed on exception

        if stage_failed:
            failed_stages_info[stage_name] = {"reason": "Execution error or missing output.", "details": str(e) if 'e' in locals() else "Function returned None or output mapping failed."}
            # Decide on error handling: stop workflow or continue? Current: continue.
            # logger.error(f"Workflow execution will continue if possible, but stage '{stage_name}' failed.")

    logger.info(f"Workflow '{workflow_name}' finished for input: {input_audio_file}")
    logger.info(f"Final workflow context keys: {list(workflow_context.keys())}")
    
    # Save final workflow context (summary of outputs/errors) to a JSON file
    final_context_path = main_run_dir / f"{run_id}_workflow_summary.json"
    try:
        with open(final_context_path, 'w') as f_ctx:
            # Convert Path objects to strings for JSON serialization
            serializable_context = {}
            for k, v_stage_output in workflow_context.items():
                if isinstance(v_stage_output, dict):
                    serializable_context[k] = {
                        sk: str(sv) if isinstance(sv, Path) else sv 
                        for sk, sv in v_stage_output.items()
                    }
                elif isinstance(v_stage_output, Path):
                     serializable_context[k] = str(v_stage_output)
                else:
                     serializable_context[k] = v_stage_output
            json.dump(serializable_context, f_ctx, indent=4, default=str) # default=str for other non-serializable types
        logger.info(f"Workflow summary saved to: {final_context_path}")
    except Exception as e_json:
        logger.error(f"Failed to save workflow summary JSON: {e_json}")

    if failed_stages_info: # Log summary of failed stages
        logger.warning(f"Workflow for {input_audio_file} completed with {len(failed_stages_info)} failed stage(s): {', '.join(failed_stages_info.keys())}")
        for stage_name, reason in failed_stages_info.items():
            logger.warning(f"  - Stage '{stage_name}' failed: {reason}")
        return False # Indicate partial or full failure
    return True # Indicate success

def segment_audio_by_clap_annotations(
    normalized_audio_path: Path,
    clap_json_path: Path,
    output_dir: Path,
    min_segment_duration: float = 5.0
) -> list:
    """
    Segments the normalized audio using Instrumental detections from the CLAP annotation JSON.
    Returns a list of output segment file paths.
    """
    import subprocess
    import json
    import math
    logger = logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(clap_json_path, 'r') as f:
        clap_data = json.load(f)
    detections = clap_data.get('detections', [])
    segments = []
    for idx, det in enumerate(detections):
        start = float(det.get('start_time_s', 0))
        end = float(det.get('end_time_s', 0))
        if end - start < min_segment_duration:
            continue
        segment_path = output_dir / f"segment_{idx+1:03d}_{math.floor(start)}s_{math.floor(end)}s.wav"
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', str(normalized_audio_path),
            '-ss', str(start), '-to', str(end),
            '-c', 'copy', str(segment_path)
        ]
        logger.info(f"Cutting segment {idx+1}: {start}s to {end}s -> {segment_path}")
        subprocess.run(ffmpeg_cmd, check=False)
        if segment_path.exists():
            segments.append(str(segment_path))
    logger.info(f"Created {len(segments)} CLAP-based segments in {output_dir}")
    return segments

def extract_audio_metadata(input_path: Path) -> dict:
    """Extracts audio metadata using ffprobe."""
    import subprocess
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries',
        'format=bit_rate,duration,format_name',
        '-show_streams', '-of', 'json', str(input_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {}
    data = json.loads(result.stdout)
    # Extract relevant fields
    info = {
        'bitrate': None,
        'channels': None,
        'sample_rate': None,
        'codec': None,
        'duration': None,
        'format': None
    }
    fmt = data.get('format', {})
    info['bitrate'] = int(fmt.get('bit_rate', 0)) // 1000 if fmt.get('bit_rate') else None
    info['duration'] = float(fmt.get('duration', 0))
    info['format'] = fmt.get('format_name')
    # Use first audio stream
    for stream in data.get('streams', []):
        if stream.get('codec_type') == 'audio':
            info['channels'] = stream.get('channels')
            info['sample_rate'] = int(stream.get('sample_rate', 0))
            info['codec'] = stream.get('codec_name')
            break
    return info

def download_audio(url, output_dir, force_redownload=True):
    """Download audio from a URL and save it to the output directory with a unique filename."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': False,
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s')
    }
    if force_redownload:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ydl_opts['outtmpl'] = os.path.join(output_dir, f'%(title)s_{timestamp}.%(ext)s')
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_file = os.path.join(output_dir, f"{info['title']}.wav")
            if not os.path.exists(downloaded_file):
                for file in os.listdir(output_dir):
                    if info['title'] in file:
                        downloaded_file = os.path.join(output_dir, file)
                        break
            logger.info(f"Downloaded audio saved to {downloaded_file}")
            return downloaded_file
    except Exception as e:
        logger.error(f"Error downloading audio from {url}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Execute an audio processing workflow from a JSON definition.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the workflow JSON configuration file.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_file", "--input_audio_file", dest="input_audio_file", type=str, help="Path to a single input audio file.")
    input_group.add_argument("--input_dir", type=str, help="Path to a directory of input audio files or call folders.")
    input_group.add_argument("--url", type=str, help="URL to download audio from.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory for workflow results.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level.")
    parser.add_argument("--cp_llm_model_id", type=str, default=None, help="LLM model ID for call_processor.")
    parser.add_argument("--final_call_output_dir", type=str, default=None, help="(Deprecated) Final output directory for processed calls.")
    parser.add_argument("--enable_clap_separation", action="store_true", help="Enable CLAP-based call segmentation after annotation. If set, audio will be segmented using CLAP annotations (e.g., telephone ringing/hang-up) and each segment will be processed as a separate call.")
    parser.add_argument("--num_speakers", type=int, default=None, help="Hard number of speakers for diarization. If set, overrides workflow config and disables automatic detection.")
    parser.add_argument("--asr_engine", type=str, choices=["whisper", "parakeet"], default=None, help="Override ASR engine for transcription stage (whisper or parakeet).")
    args = parser.parse_args()

    # Initial logging setup (console only until run-specific log file is set in execute_workflow)
    # setup_workflow_logging(args.log_level) # This will be called again in execute_workflow with file handler
    # Simplified initial setup, execute_workflow will set up its own file-based logging per run.
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format='%(asctime)s [%(levelname)s] - %(name)s.%(funcName)s - %(message)s',
                        handlers=[logging.StreamHandler()])

    files_to_process = []
    is_potential_call_folder = False
    if args.url:
        download_target_dir = os.path.join(args.output_dir, "downloads")
        os.makedirs(download_target_dir, exist_ok=True)
        try:
            logger.info(f"Downloading audio from {args.url} to {download_target_dir}")
            downloaded_file = download_audio(args.url, download_target_dir)
            if not downloaded_file or not os.path.exists(downloaded_file):
                raise ValueError("download_audio failed to return a valid path.")
            files_to_process.append(downloaded_file)
        except Exception as e:
            logger.error(f"Failed to download URL {args.url}: {e}")
            return
    elif args.input_audio_file:
        files_to_process.append(args.input_audio_file)
    elif args.input_dir:
        input_dir_path = Path(args.input_dir)
        if not input_dir_path.is_dir():
            logger.error(f"Input directory not found: {args.input_dir}")
            return
        
        all_items_in_dir = os.listdir(args.input_dir) # Get all items once to avoid multiple listdir calls

        # First pass: Determine if the directory is a potential call folder
        for item_name in all_items_in_dir:
            if os.path.isfile(input_dir_path / item_name):
                if item_name.lower().startswith(("recv_out", "trans_out")):
                    is_potential_call_folder = True
                    logger.info(f"Directory '{args.input_dir}' identified as a potential call folder due to presence of 'recv_out' or 'trans_out' files. Specific filtering will apply for 'out-*' files.")
                    break # Found one, no need to check further for this flag

        logger.info(f"Scanning directory '{args.input_dir}' for files to process through the main workflow...")
        supported_audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        for filename in all_items_in_dir:
            file_path = input_dir_path / filename
            filename_lower = filename.lower()

            if file_path.is_file():
                # Apply skipping logic if it's a potential call folder and file matches "out-*" pattern
                if is_potential_call_folder:
                    if filename_lower.startswith("out-") and \
                       not filename_lower.startswith("recv_out-") and \
                       not filename_lower.startswith("trans_out-"):
                        logger.info(f"Skipping general 'out-' prefixed file in call folder: {filename} from main workflow processing.")
                        continue # Skip this file from being added to files_to_process

                # Existing audio file extension check for files not skipped
                if file_path.suffix.lower() in supported_audio_extensions:
                    files_to_process.append(str(file_path))
                else:
                    logger.debug(f"Skipping non-audio or already explicitly skipped item: {filename}")
            else:
                logger.debug(f"Skipping non-file item: {filename}")
    
    if not files_to_process:
        logger.info("No audio files found to process.")
        return

    logger.info(f"Found {len(files_to_process)} audio file(s) to process.")
    all_workflows_succeeded = True
    processed_count = 0

    if args.enable_clap_separation:
        print("[INFO] CLAP-based call segmentation is ENABLED. After annotation, audio will be segmented and each segment processed as a call.")
        # --- CLAP-based segmentation logic ---
        # 1. Locate the annotation output (e.g., 00_clap_event_annotation/annotations.json)
        # 2. Locate the normalized audio (e.g., 01_audio_preprocessing/yourfile_normalized.wav)
        # 3. Segment and output to 06_clap_separation/calls/
        # 4. Add segments to files_to_process
        new_segments = []
        for audio_file_str in list(files_to_process):
            audio_file = Path(audio_file_str)
            # Find the workflow run directory for this file (if it exists)
            workflow_run_dirs = list(Path(args.output_dir).glob(f"*{audio_file.stem}*"))
            if not workflow_run_dirs:
                logger.warning(f"No workflow run directory found for {audio_file.name} in {args.output_dir}")
                continue
            workflow_run_dir = workflow_run_dirs[0]
            clap_dir = workflow_run_dir / "00_clap_event_annotation"
            norm_dir = workflow_run_dir / "01_audio_preprocessing"
            # Find annotation JSON and normalized audio
            clap_json = None
            for f in clap_dir.glob("*.json"):
                if "instrumental" in f.name.lower():
                    clap_json = f
                    break
            if not clap_json:
                logger.warning(f"No CLAP annotation JSON found in {clap_dir}")
                continue
            norm_audio = None
            for f in norm_dir.glob("*normalized*.wav"):
                norm_audio = f
                break
            if not norm_audio:
                logger.warning(f"No normalized audio found in {norm_dir}")
                continue
            # Output dir for segments
            seg_dir = workflow_run_dir / "06_clap_separation" / "calls"
            segments = segment_audio_by_clap_annotations(norm_audio, clap_json, seg_dir)
            new_segments.extend(segments)
        # Replace files_to_process with segments if any were found
        if new_segments:
            files_to_process = new_segments
    else:
        print("[INFO] CLAP-based call segmentation is DISABLED.")

    metadata_map = {}
    for i, audio_file_str in enumerate(files_to_process):
        processed_count += 1
        audio_file = Path(audio_file_str)
        logger.info(f"--- Processing file {processed_count}/{len(files_to_process)}: {audio_file} ---")
        meta = extract_audio_metadata(audio_file)
        meta_path = audio_file.with_suffix('.metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        metadata_map[str(audio_file)] = meta
        logger.info(f"Extracted metadata for {audio_file.name}: {meta}")
        # Apollo restoration logic
        run_apollo = False
        if meta.get('format', '').lower() == 'mp3':
            br = meta.get('bitrate', 0)
            ch = meta.get('channels', 0)
            if (br <= 96 and ch == 2) or (br <= 64 and ch == 1):
                run_apollo = True
        if run_apollo:
            logger.info(f"Running Apollo restoration for {audio_file.name} (bitrate={meta.get('bitrate')}kbps, channels={meta.get('channels')})")
            apollo_out = audio_file.with_name(audio_file.stem + '_apollo_restored.wav')
            apollo_cmd = [
                'python', 'Apollo/inference.py',
                '--in_wav', str(audio_file),
                '--out_wav', str(apollo_out)
            ]
            subprocess.run(apollo_cmd, check=False)
            if apollo_out.exists():
                files_to_process[i] = str(apollo_out)
                meta['apollo_restored'] = True
                logger.info(f"Apollo restoration complete: {apollo_out}")
            else:
                logger.warning(f"Apollo restoration failed for {audio_file.name}")
        else:
            meta['apollo_restored'] = False
        # Update metadata file
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        # If --num_speakers is set, override the diarization stage config before running the workflow
        if args.num_speakers is not None or args.asr_engine is not None:
            # Load the workflow config file
            with open(args.config_file, 'r') as f:
                workflow_config = json.load(f)
            # Find the diarization stage and override num_speakers
            for stage in workflow_config.get('stages', []):
                if args.num_speakers is not None and stage.get('stage_name', '').lower() == 'speaker_diarization':
                    if 'config' not in stage:
                        stage['config'] = {}
                    stage['config']['num_speakers'] = args.num_speakers
                if args.asr_engine is not None and stage.get('stage_name', '').lower() == 'transcription':
                    if 'config' not in stage:
                        stage['config'] = {}
                    stage['config']['asr_engine'] = args.asr_engine
                    logger.info(f"Overriding transcription ASR engine to '{args.asr_engine}' via CLI.")
            # Save the modified workflow config to a temp file
            import tempfile
            with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json') as tmpf:
                json.dump(workflow_config, tmpf, indent=2)
                tmp_config_path = tmpf.name
            args.config_file = tmp_config_path
            if args.num_speakers is not None:
                logger.info(f"Overriding diarization num_speakers to {args.num_speakers} via CLI. Using temp config: {tmp_config_path}")
        success = execute_workflow(args.config_file, audio_file_str, args.output_dir, args.log_level)
        if not success:
            all_workflows_succeeded = False
            logger.error(f"Workflow execution failed for: {audio_file}")
            # Decide if we should continue with other files or stop.
            # For now, continue processing other files.
        logger.info(f"--- Finished processing file: {audio_file} ---")

    logger.info(f"All individual file workflow executions complete. Overall success: {all_workflows_succeeded}")

    # --- Always run the final output builder as the last step ---
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / '05_final_output'
    output_dir.mkdir(parents=True, exist_ok=True)
    build_final_output(project_root, output_dir)

if __name__ == "__main__":
    main() 