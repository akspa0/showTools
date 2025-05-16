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

            # Get the configuration specific to the function
            function_specific_config = stage_def.get("config", {})
            
            # Prepare arguments for the function call
            # Some functions might expect all their params directly, 
            # others might expect a 'config' dict plus specific inputs.
            # We'll pass resolved_inputs directly, and the function_specific_config as 'config' if present.
            # The stage function must be designed to accept these.
            
            args_for_function = {**resolved_inputs} # Start with resolved inputs

            # Add the PII-safe identifier stem as a standard argument for stages to use in naming temp/output files
            args_for_function['pii_safe_file_prefix'] = pii_safe_identifier_stem 

            # If the stage_def.config is not empty, pass it as a 'config' kwarg to the function.
            # The called function should expect a 'config' argument if this is provided.
            # Alternatively, one could merge stage_def.config into args_for_function if functions
            # are designed to take all their parameters flatly. This current approach is more explicit.
            if function_specific_config:
                 args_for_function['config'] = function_specific_config


            logger.debug(f"Importing module: {module_name}, function: {function_name}")
            module_to_import = importlib.import_module(module_name)
            function_to_call = getattr(module_to_import, function_name)
            
            logger.info(f"Executing: {module_name}.{function_name}")
            logger.debug(f"With arguments (first level): { {k: type(v) for k, v in args_for_function.items()} }")


            stage_result = function_to_call(**args_for_function)
            
            # Store results in context
            if stage_def.get("outputs"):
                for context_key, result_key_expression in stage_def["outputs"].items():
                    if result_key_expression == "return_value":
                        if stage_name not in workflow_context or not isinstance(workflow_context.get(stage_name), dict):
                            workflow_context[stage_name] = {}
                        workflow_context[stage_name][context_key] = stage_result # Store the result under the context_key
                        logger.debug(f"Stage '{stage_name}' full result stored in context['{stage_name}']['{context_key}'].")
                    elif isinstance(stage_result, dict) and result_key_expression in stage_result:
                        # This branch handles "context_key": "key_directly_in_result_dict"
                        # This is likely not used often if "return_value[key]" is preferred.
                        if stage_name not in workflow_context or not isinstance(workflow_context.get(stage_name), dict):
                             workflow_context[stage_name] = {}
                        workflow_context[stage_name][context_key] = stage_result[result_key_expression]
                        logger.debug(f"Stage '{stage_name}' output '{result_key_expression}' (direct key) stored in context['{stage_name}']['{context_key}'].")
                    elif isinstance(stage_result, dict) and result_key_expression.startswith("return_value[") and result_key_expression.endswith("]"):
                        actual_key = result_key_expression[len("return_value["):-1]
                        if actual_key in stage_result:
                            # Store the output under the stage_name, then the context_key from workflow
                            if stage_name not in workflow_context or not isinstance(workflow_context.get(stage_name), dict):
                                workflow_context[stage_name] = {}
                            workflow_context[stage_name][context_key] = stage_result[actual_key]
                            logger.debug(f"Stage '{stage_name}' output '{actual_key}' from result dict stored in context['{stage_name}']['{context_key}'].")
                        else:
                            logger.error(f"Output key '{actual_key}' (from expression '{result_key_expression}') not found in stage '{stage_name}' result dict. Stage considered failed.")
                            stage_failed = True # Mark stage as failed
                    else:
                        logger.error(f"Output key expression '{result_key_expression}' for context key '{context_key}' could not be resolved from stage '{stage_name}' result. Stage considered failed.")
                        stage_failed = True # Mark stage as failed
            
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

def main():
    parser = argparse.ArgumentParser(description="Execute an audio processing workflow from a JSON definition.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the workflow JSON configuration file.")
    
    # Input can be a single file or a directory
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_audio_file", type=str, help="Path to a single input audio file.")
    input_group.add_argument("--input_dir", type=str, help="Path to a directory of audio files to process.")
    
    parser.add_argument("--output_dir", type=str, required=True, help="Base directory for workflow run outputs.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Logging level.")

    # New arguments for call_processor integration
    parser.add_argument(
        "--final_call_output_dir",
        type=str,
        default=None, # Optional
        help="Base directory to save final combined call outputs. If provided and input_dir "
             "contains recv_out/trans_out files, call processing will be triggered after "
             "all individual files are processed."
    )
    parser.add_argument(
        "--cp_llm_model_id",
        type=str,
        default=None, # Optional
        help="LLM model identifier to be used by call_processor.py for combined call summaries. "
             "Example: 'NousResearch/Hermes-2-Pro-Llama-3-8B'"
    )

    args = parser.parse_args()

    # Initial logging setup (console only until run-specific log file is set in execute_workflow)
    # setup_workflow_logging(args.log_level) # This will be called again in execute_workflow with file handler
    # Simplified initial setup, execute_workflow will set up its own file-based logging per run.
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format='%(asctime)s [%(levelname)s] - %(name)s.%(funcName)s - %(message)s',
                        handlers=[logging.StreamHandler()])

    files_to_process = []
    is_potential_call_folder = False # Flag to indicate if input_dir contains call-specific files

    if args.input_audio_file:
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

    for audio_file_str in files_to_process:
        processed_count += 1
        logger.info(f"--- Processing file {processed_count}/{len(files_to_process)}: {audio_file_str} ---")
        success = execute_workflow(args.config_file, audio_file_str, args.output_dir, args.log_level)
        if not success:
            all_workflows_succeeded = False
            logger.error(f"Workflow execution failed for: {audio_file_str}")
            # Decide if we should continue with other files or stop.
            # For now, continue processing other files.
        logger.info(f"--- Finished processing file: {audio_file_str} ---")

    logger.info(f"All individual file workflow executions complete. Overall success: {all_workflows_succeeded}")

    # --- Conditional call_processor.py invocation --- 
    if args.input_dir and args.output_dir and args.final_call_output_dir:
        if not all_workflows_succeeded:
            logger.warning("One or more individual file workflows failed. Skipping combined call processing.")
            return # Return instead of exiting to allow potential cleanup or further script logic if any
            
        # Use the is_potential_call_folder flag determined earlier
        if is_potential_call_folder:
            logger.info(f"Call folder detected in '{args.input_dir}'. Attempting to run call_processor.py.")
            
            # Determine path to call_processor.py (assuming it's in the same directory as workflow_executor.py)
            current_script_dir = Path(__file__).resolve().parent
            path_to_call_processor_script = current_script_dir / "call_processor.py"

            if not path_to_call_processor_script.is_file():
                logger.error(f"call_processor.py not found at expected location: {path_to_call_processor_script}. Cannot proceed with call processing.")
                return

            call_processor_cmd_list = [
                "python", str(path_to_call_processor_script),
                "--input_run_dir", str(Path(args.output_dir).resolve()), 
                "--output_call_dir", str(Path(args.final_call_output_dir).resolve()),
                "--log_level", args.log_level 
            ]
            if args.cp_llm_model_id:
                call_processor_cmd_list.extend(["--llm_model_id", args.cp_llm_model_id])
            
            try:
                logger.info(f"Executing call_processor: {' '.join(call_processor_cmd_list)}")
                process_env = os.environ.copy()
                
                completed_process = subprocess.run(
                    call_processor_cmd_list, 
                    capture_output=True, 
                    text=True, 
                    check=False, # Do not raise exception on non-zero exit code
                    env=process_env,
                    # cwd=str(current_script_dir) # Optional: Set CWD for the subprocess if needed
                )
                
                if completed_process.returncode == 0:
                    logger.info("call_processor.py executed successfully.")
                    if completed_process.stdout:
                        logger.debug(f"call_processor.py stdout:\n{completed_process.stdout}")
                    if completed_process.stderr: # Log stderr even on success, as it might contain warnings
                        logger.debug(f"call_processor.py stderr:\n{completed_process.stderr}")
                else:
                    logger.error(f"call_processor.py failed with return code {completed_process.returncode}")
                    if completed_process.stdout:
                        logger.error(f"call_processor.py stdout:\n{completed_process.stdout}")
                    if completed_process.stderr:
                        logger.error(f"call_processor.py stderr:\n{completed_process.stderr}")
            except FileNotFoundError: 
                logger.error(f"Failed to run call_processor.py: 'python' command or script '{path_to_call_processor_script}' not found. Ensure Python is in PATH and script exists.")
            except Exception as e:
                logger.error(f"An unexpected error occurred while trying to run call_processor.py: {e}", exc_info=True)
        else:
            # This 'else' branch is for when --final_call_output_dir is given, but --input_dir did not contain call-like files
            if args.final_call_output_dir: # Ensure this log only appears if the user intended call processing
                 logger.info(f"Input directory '{args.input_dir}' does not appear to be a call folder (no 'recv_out' or 'trans_out' files found). Skipping combined call processing step.")
    else:
        if args.input_dir and not args.final_call_output_dir:
            logger.info("Processed all files in input_dir. --final_call_output_dir not specified, so skipping combined call processing step.")
        elif not args.input_dir and args.final_call_output_dir:
            logger.info("--final_call_output_dir was specified, but --input_dir was not. Combined call processing is only triggered when processing an input directory.")

if __name__ == "__main__":
    main() 