import os
import re
import logging
import argparse
import shutil
import subprocess # Added for ffmpeg
import json # Added for transcript merging
from pathlib import Path
from datetime import datetime # Will be needed for parsing timestamps if not in filenames

# Attempt to import the LLM module function.
# This is a placeholder and might need adjustment based on actual project structure and PYTHONPATH.
# If llm_module.py is in the same directory, this should work.
try:
    from llm_module import generate_llm_summary # Assuming this function exists
except ImportError:
    # Fallback if the import fails, so the script can still be parsed.
    # Actual LLM calls will not work without this module.
    def generate_llm_summary(*args, **kwargs):
        logger.error("llm_module.generate_llm_summary could not be imported. LLM summarization will be skipped.")
        return None
    logger.warning("Could not import 'generate_llm_summary' from 'llm_module'. Combined summary generation will be disabled.")


# Global logger for the processor
logger = logging.getLogger(__name__)

# Define constants for stage keywords and filenames
VOCAL_STEM_FILENAME = "vocals_normalized.wav"
TRANSCRIPT_FILENAME = "transcript.json"
SUMMARY_FILENAME = "final_analysis_summary.txt" # Individual stream summary
MIXED_VOCALS_FILENAME = "mixed_vocals.wav"
MERGED_TRANSCRIPT_FILENAME = "merged_transcript.json"
COMBINED_SUMMARY_FILENAME = "combined_call_summary.txt" # For the combined call

AUDIO_PREPROCESSING_STAGE_KEYWORD = "audio_preprocessing"
TRANSCRIPTION_STAGE_KEYWORD = "transcription"
LLM_SUMMARY_STAGE_KEYWORD = "llm_summary_and_analysis" # Matches "04_llm_summary_and_analysis"

# Pattern for speaker directories (e.g., S0, S1, SPEAKER_00)
SPEAKER_DIR_PATTERN = re.compile(r"^(S\d+|SPEAKER_\d+)$")

def setup_logging(log_level_str='INFO'):
    """Configures logging for the application."""
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logging.basicConfig(level=log_level, 
                        format='%(asctime)s [%(levelname)s] - %(name)s.%(funcName)s - %(message)s',
                        handlers=[logging.StreamHandler()])
    # Suppress overly verbose logs from libraries if needed
    # logging.getLogger("some_library").setLevel(logging.WARNING)
    logger.info(f"Logging setup with level: {log_level_str}")

def generate_call_id_from_workflow_dir_name(dir_name: str):
    """
    Extracts a potential call_id from a workflow executor output directory name.
    Example dir_name: DefaultAudioAnalysisWorkflow_recv_out-188855512121-725-20250512-020958-1747015798.308_20250515_103000
    We want to extract something like: call_20250512-020958
    This is an adaptation of the logic in audiotoolkit_phase1.py's generate_call_id.
    """
    # First, try to get the original filename part
    # The dir name is typically WorkflowName_OriginalFileNameStem_Timestamp
    match_original_stem = re.match(r".*?_(recv_out.*?|trans_out.*?)_\d{8}_\d{6}", dir_name)
    if not match_original_stem:
        # Fallback or alternative pattern if the above is too strict or changes
        # This might be if workflow name itself has underscores. Let's try to find the timestamp directly from original part.
        # Look for original filename patterns directly if the first regex fails
        # Often workflow_dir name is like: WORKFLOWNAME_ORIGINALFILESTEM_RUNDATETIME
        # Example: DefaultAudioAnalysisWorkflow_recv_out-188855512121-725-20250512-020958-1747015798.308_20240515_120000
        # The part we are interested in is 'recv_out-188855512121-725-20250512-020958-1747015798.308'
        
        # Attempt to extract known prefixes and the timestamp part from them.
        # This regex is designed to find "recv_out" or "trans_out" followed by the specific timestamp format.
        greedy_match = re.search(r"(recv_out-[\d-]+-(\d{8}-\d{6})-\d+\.\d+)", dir_name)
        if not greedy_match:
            greedy_match = re.search(r"(trans_out-[\d-]+-(\d{8}-\d{6})-\d+\.\d+)", dir_name)

        if greedy_match:
            timestamp_part = greedy_match.group(2) # The YYYYMMDD-HHMMSS part
            return f"call_{timestamp_part}"
        else:
            # If no timestamp match from original-like stem, use a generic ID from the dir name
            # This is a last resort and might not group calls correctly if timestamps are missing.
            sanitized_dir_name = re.sub(r'[^a-zA-Z0-9_-]', '', dir_name)
            logger.warning(f"Could not extract timestamp-based call_id from '{dir_name}\'. Using sanitized dir name as part of ID.")
            # Try to find any YYYYMMDD-HHMMSS pattern in the whole dir_name as a last effort for a date.
            date_match_anywhere = re.search(r"(\d{8}-\d{6})", dir_name)
            if date_match_anywhere:
                return f"call_{date_match_anywhere.group(1)}_{sanitized_dir_name[:30]}" # less ideal
            return f"call_unknown_{sanitized_dir_name[:50]}" # even less ideal

    original_filename_stem = match_original_stem.group(1)

    # Try to extract a timestamp like YYYYMMDD-HHMMSS from the original_filename_stem
    match_timestamp = re.search(r'(\d{8}-\d{6})-\d{10}(?:\.\d+)?', original_filename_stem)
    if match_timestamp:
        return f"call_{match_timestamp.group(1)}"
    else:
        # Fallback if specific timestamp pattern not found in the stem, but we have the stem
        sanitized_stem = re.sub(r'[^a-zA-Z0-9_-]', '', original_filename_stem)
        logger.warning(f"Could not extract specific YYYYMMDD-HHMMSS timestamp from stem '{original_filename_stem}\' in dir '{dir_name}\'. Using sanitized stem for ID.")
        return f"call_{sanitized_stem[:50]}"


def find_call_pairs(input_run_dir: Path):
    """
    Scans the input_run_dir for workflow output directories, generates call_ids,
    and groups them into pairs (recv_out, trans_out) or singles.
    
    Returns:
        dict: {
            call_id_1: {"recv_run_dir": Path(), "trans_run_dir": Path(), "type": "pair"},
            call_id_2: {"run_dir": Path(), "type": "single_recv" or "single_trans" or "single_unknown"},
            ...
        }
    """
    logger.info(f"Scanning for call pairs in: {input_run_dir}")
    potential_calls = {} # Stores call_id -> list of full_path_to_workflow_run_dir

    if not input_run_dir.is_dir():
        logger.error(f"Input run directory not found or not a directory: {input_run_dir}")
        return {}

    for item in input_run_dir.iterdir():
        if item.is_dir(): # Each item is a workflow run directory
            dir_name = item.name
            call_id = generate_call_id_from_workflow_dir_name(dir_name)
            
            if call_id not in potential_calls:
                potential_calls[call_id] = []
            potential_calls[call_id].append(item) # Store the Path object
            logger.debug(f"Associated dir '{dir_name}' with call_id '{call_id}'.")

    # Process potential_calls to identify definitive pairs and singles
    call_jobs = {}
    for call_id, run_dirs in potential_calls.items():
        recv_dir = None
        trans_dir = None
        
        for run_dir in run_dirs:
            dir_name = run_dir.name.lower() # Use lower for matching prefixes
            if "recv_out" in dir_name:
                if recv_dir: logger.warning(f"Multiple recv_out dirs found for call_id '{call_id}\': {recv_dir.name}, {run_dir.name}. Using first one.")
                else: recv_dir = run_dir
            elif "trans_out" in dir_name:
                if trans_dir: logger.warning(f"Multiple trans_out dirs found for call_id '{call_id}\': {trans_dir.name}, {run_dir.name}. Using first one.")
                else: trans_dir = run_dir
        
        if recv_dir and trans_dir:
            call_jobs[call_id] = {"recv_run_dir": recv_dir, "trans_run_dir": trans_dir, "type": "pair"}
            logger.info(f"Identified PAIR for call_id '{call_id}\': recv='{recv_dir.name}\', trans='{trans_dir.name}\'")
        elif recv_dir:
            call_jobs[call_id] = {"run_dir": recv_dir, "type": "single_recv"}
            logger.info(f"Identified SINGLE_RECV for call_id '{call_id}\': '{recv_dir.name}\'")
        elif trans_dir:
            call_jobs[call_id] = {"run_dir": trans_dir, "type": "single_trans"}
            logger.info(f"Identified SINGLE_TRANS for call_id '{call_id}\': '{trans_dir.name}\'")
        else:
            # This case might happen if a directory didn't match recv_out/trans_out but got a call_id
            # Or if multiple non-recv/trans dirs got the same call_id.
            if run_dirs: # If there are any dirs associated
                # Take the first one as a representative for a single unknown type
                call_jobs[call_id] = {"run_dir": run_dirs[0], "type": "single_unknown"}
                logger.warning(f"Identified SINGLE_UNKNOWN for call_id '{call_id}\' from dir '{run_dirs[0].name}\'. Contains neither 'recv_out' nor 'trans_out' in name.")
            else: # Should not happen if potential_calls was populated correctly
                logger.error(f"Call ID '{call_id}\' had no associated directories after prefix checking. This is unexpected.")


    return call_jobs


def _find_output_file(workflow_run_dir: Path, stage_keyword: str, filename: str, call_id_for_logging: str = "unknown_call") -> Path | None:
    """
    Finds a specific output file within a stage subdirectory of a workflow run directory.
    The stage subdirectory is identified by a keyword.
    """
    if not workflow_run_dir or not workflow_run_dir.is_dir():
        logger.warning(f"[{call_id_for_logging}] Workflow run directory '{workflow_run_dir}' is invalid or not found for finding '{filename}'.")
        return None

    found_stage_dir = None
    for item in workflow_run_dir.iterdir():
        if item.is_dir() and stage_keyword in item.name:
            found_stage_dir = item
            break # Assume first match is the correct one

    if not found_stage_dir:
        logger.warning(f"[{call_id_for_logging}] No stage directory found containing keyword '{stage_keyword}' in '{workflow_run_dir}' while looking for '{filename}'.")
        return None

    target_file = found_stage_dir / filename
    if target_file.exists() and target_file.is_file():
        logger.debug(f"[{call_id_for_logging}] Found '{filename}' at '{target_file}'.")
        return target_file
    else:
        logger.warning(f"[{call_id_for_logging}] File '{filename}' not found in stage directory '{found_stage_dir.name}' (path: '{target_file}').")
        return None


def _mix_stereo_vocals(recv_vocal_path: Path, trans_vocal_path: Path, output_mixed_vocal_path: Path, call_id_for_logging: str) -> bool:
    """
    Mixes two mono vocal WAV files (recv and trans) into a single stereo WAV file using ffmpeg.
    Recv stream will be on the left channel, Trans stream on the right.
    Returns True if mixing was successful, False otherwise.
    """
    if not (recv_vocal_path.exists() and trans_vocal_path.exists()):
        logger.error(f"[{call_id_for_logging}] One or both vocal stems not found for mixing: RECV: '{recv_vocal_path}', TRANS: '{trans_vocal_path}'")
        return False

    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # Overwrite output files without asking
        '-i', str(recv_vocal_path),
        '-i', str(trans_vocal_path),
        '-filter_complex', "[0:a][1:a]join=inputs=2:channel_layout=stereo[a]",
        '-map', '[a]',
        str(output_mixed_vocal_path)
    ]

    logger.debug(f"[{call_id_for_logging}] Executing ffmpeg command for stereo mixing: {' '.join(ffmpeg_cmd)}")
    try:
        process = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)
        if process.returncode == 0:
            logger.info(f"[{call_id_for_logging}] Successfully mixed vocal stems to '{output_mixed_vocal_path}'.")
            return True
        else:
            logger.error(f"[{call_id_for_logging}] ffmpeg mixing failed for '{output_mixed_vocal_path}'. Return code: {process.returncode}")
            logger.error(f"[{call_id_for_logging}] ffmpeg stderr: {process.stderr}")
            logger.error(f"[{call_id_for_logging}] ffmpeg stdout: {process.stdout}")
            return False
    except FileNotFoundError:
        logger.error(f"[{call_id_for_logging}] ffmpeg command not found. Please ensure ffmpeg is installed and in your PATH.")
        return False
    except Exception as e:
        logger.error(f"[{call_id_for_logging}] An unexpected error occurred during ffmpeg execution for '{output_mixed_vocal_path}': {e}")
        return False


def _copy_soundbite_directories(
    workflow_run_dir: Path, 
    transcription_stage_keyword: str, 
    call_output_dir: Path, 
    stream_prefix: str, 
    call_id_for_logging: str
) -> bool:
    """
    Copies speaker-specific soundbite directories (e.g., S0, S1) from the transcription stage
    of a workflow run directory to the final call output directory.
    The speaker directories in the destination will be prefixed (e.g., RECV_S0).
    """
    if not workflow_run_dir or not workflow_run_dir.is_dir():
        logger.warning(f"[{call_id_for_logging}] Workflow run directory '{workflow_run_dir}' is invalid or not found for copying soundbites.")
        return False

    transcription_stage_dir = None
    for item in workflow_run_dir.iterdir():
        if item.is_dir() and transcription_stage_keyword in item.name.lower(): # Ensure case-insensitive keyword check
            transcription_stage_dir = item
            break
    
    if not transcription_stage_dir:
        logger.warning(f"[{call_id_for_logging}] No transcription stage directory found containing keyword '{transcription_stage_keyword}' in '{workflow_run_dir}'. Cannot copy soundbites.")
        return False

    logger.info(f"[{call_id_for_logging}] Found transcription stage directory: {transcription_stage_dir}. Searching for speaker soundbite directories.")
    
    copied_any = False
    for speaker_dir in transcription_stage_dir.iterdir():
        if speaker_dir.is_dir() and SPEAKER_DIR_PATTERN.match(speaker_dir.name):
            source_speaker_dir_path = speaker_dir
            dest_speaker_dir_name = f"{stream_prefix}_{speaker_dir.name}"
            dest_speaker_dir_path = call_output_dir / dest_speaker_dir_name
            
            try:
                shutil.copytree(source_speaker_dir_path, dest_speaker_dir_path, dirs_exist_ok=True)
                logger.info(f"[{call_id_for_logging}] Successfully copied soundbite directory '{source_speaker_dir_path.name}' to '{dest_speaker_dir_path}'.")
                copied_any = True
            except Exception as e:
                logger.error(f"[{call_id_for_logging}] Error copying soundbite directory from '{source_speaker_dir_path}' to '{dest_speaker_dir_path}': {e}")
    
    if not copied_any:
        logger.warning(f"[{call_id_for_logging}] No speaker soundbite directories (e.g., S0, S1) found or copied from '{transcription_stage_dir}'.")
        
    return copied_any


def _merge_transcripts(recv_transcript_path: Path, trans_transcript_path: Path, output_merged_transcript_path: Path, call_id_for_logging: str) -> bool:
    """
    Merges two transcript JSON files (recv and trans) into a single chronological transcript.
    Speaker labels are prefixed with RECV_ or TRANS_.
    Returns True if merging was successful, False otherwise.
    """
    merged_segments = []

    try:
        # Process RECV transcript
        if not recv_transcript_path.exists():
            logger.warning(f"[{call_id_for_logging}] RECV transcript not found at '{recv_transcript_path}' for merging.")
            # Decide if this is a hard fail or if we can proceed with only trans
            # For now, let's assume both are needed or we create an empty/partial merge
            return False # Or adapt to merge only one if that's desired
        with open(recv_transcript_path, 'r', encoding='utf-8') as f:
            recv_data = json.load(f)
        
        # Assuming recv_data is a list of segments or a dict with a key like 'segments'
        # For now, assume it's directly a list of segments as per Whisper output structure
        if isinstance(recv_data, list):
            for segment in recv_data:
                segment['speaker'] = f"RECV_{segment.get('speaker', 'UNKNOWN')}"
                merged_segments.append(segment)
        else:
            logger.warning(f"[{call_id_for_logging}] RECV transcript data at '{recv_transcript_path}' is not a list as expected.")
            # Handle cases where transcript might be under a key, e.g., recv_data.get('segments', [])
            # For now, this is treated as an issue.

        # Process TRANS transcript
        if not trans_transcript_path.exists():
            logger.warning(f"[{call_id_for_logging}] TRANS transcript not found at '{trans_transcript_path}' for merging.")
            return False # Or adapt
        with open(trans_transcript_path, 'r', encoding='utf-8') as f:
            trans_data = json.load(f)

        if isinstance(trans_data, list):
            for segment in trans_data:
                segment['speaker'] = f"TRANS_{segment.get('speaker', 'UNKNOWN')}"
                merged_segments.append(segment)
        else:
            logger.warning(f"[{call_id_for_logging}] TRANS transcript data at '{trans_transcript_path}' is not a list as expected.")

    except json.JSONDecodeError as e:
        logger.error(f"[{call_id_for_logging}] Error decoding JSON for transcript merging: {e}")
        return False
    except IOError as e:
        logger.error(f"[{call_id_for_logging}] File I/O error during transcript merging: {e}")
        return False
    except Exception as e:
        logger.error(f"[{call_id_for_logging}] Unexpected error during transcript processing for merging: {e}")
        return False

    # Sort by start_time if segments were added
    if not merged_segments:
        logger.warning(f"[{call_id_for_logging}] No segments were processed for merging. Output will be empty.")
    
    try:
        merged_segments.sort(key=lambda x: x.get('start_time', 0))
    except TypeError as e:
        logger.error(f"[{call_id_for_logging}] Error sorting merged segments. Ensure 'start_time' is present and comparable: {e}")
        # Potentially write unsorted or return False
        # For now, we'll write whatever we have, possibly unsorted if start_time is missing/mixed types

    # Write merged transcript
    try:
        with open(output_merged_transcript_path, 'w', encoding='utf-8') as f:
            json.dump(merged_segments, f, indent=2, ensure_ascii=False)
        logger.info(f"[{call_id_for_logging}] Successfully merged transcripts to '{output_merged_transcript_path}'.")
        return True
    except IOError as e:
        logger.error(f"[{call_id_for_logging}] File I/O error writing merged transcript to '{output_merged_transcript_path}': {e}")
        return False
    except Exception as e:
        logger.error(f"[{call_id_for_logging}] Unexpected error writing merged transcript: {e}")
        return False


def process_call(call_id: str, job_details: dict, output_call_base_dir: Path, config: dict):
    """
    Processes a single call (pair or single).
    Retrieves necessary files from workflow run directories and performs mixing/combination.
    """
    logger.info(f"[{call_id}] Starting processing for call type: {job_details['type']}")
    call_output_dir = output_call_base_dir / call_id
    call_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Configuration from main config dict ---
    # Example: tones_config = config.get('tones', {})
    # Example: mixing_config = config.get('mixing', {})

    # --- Get run directories from job_details ---
    recv_run_dir = job_details.get("recv_run_dir")
    trans_run_dir = job_details.get("trans_run_dir")

    # --- Stage 2: Process RECV stream ---
    recv_vocal_stem_path = None
    recv_transcript_path = None
    recv_summary_path = None

    if recv_run_dir:
        logger.info(f"[{call_id}] Processing RECV stream from: {recv_run_dir.name}")
        recv_vocal_stem_path = _find_output_file(recv_run_dir, AUDIO_PREPROCESSING_STAGE_KEYWORD, VOCAL_STEM_FILENAME, call_id)
        recv_transcript_path = _find_output_file(recv_run_dir, TRANSCRIPTION_STAGE_KEYWORD, TRANSCRIPT_FILENAME, call_id)
        recv_summary_path = _find_output_file(recv_run_dir, LLM_SUMMARY_STAGE_KEYWORD, SUMMARY_FILENAME, call_id)
        
        # Copy soundbites for RECV stream
        _copy_soundbite_directories(
            workflow_run_dir=recv_run_dir,
            transcription_stage_keyword=TRANSCRIPTION_STAGE_KEYWORD.lower(),
            call_output_dir=call_output_dir,
            stream_prefix="RECV",
            call_id_for_logging=call_id
        )

        # Copy individual RECV summary if found
        if recv_summary_path and recv_summary_path.exists():
            shutil.copy2(recv_summary_path, call_output_dir / f"{call_id}_recv_{SUMMARY_FILENAME}")
            logger.info(f"[{call_id}] Copied recv summary to output directory.")

    # --- Stage 3: Process TRANS stream ---
    trans_vocal_stem_path = None
    trans_transcript_path = None
    trans_summary_path = None

    if trans_run_dir:
        logger.info(f"[{call_id}] Processing TRANS stream from: {trans_run_dir.name}")
        trans_vocal_stem_path = _find_output_file(trans_run_dir, AUDIO_PREPROCESSING_STAGE_KEYWORD, VOCAL_STEM_FILENAME, call_id)
        trans_transcript_path = _find_output_file(trans_run_dir, TRANSCRIPTION_STAGE_KEYWORD, TRANSCRIPT_FILENAME, call_id)
        trans_summary_path = _find_output_file(trans_run_dir, LLM_SUMMARY_STAGE_KEYWORD, SUMMARY_FILENAME, call_id)

        # Copy soundbites for TRANS stream
        _copy_soundbite_directories(
            workflow_run_dir=trans_run_dir,
            transcription_stage_keyword=TRANSCRIPTION_STAGE_KEYWORD.lower(),
            call_output_dir=call_output_dir,
            stream_prefix="TRANS",
            call_id_for_logging=call_id
        )

        # Copy individual TRANS summary if found
        if trans_summary_path and trans_summary_path.exists():
            shutil.copy2(trans_summary_path, call_output_dir / f"{call_id}_trans_{SUMMARY_FILENAME}")
            logger.info(f"[{call_id}] Copied trans summary to output directory.")

    # Handle single stream cases (not a pair but a single recv_out or trans_out)
    elif job_details["type"] in ["single_recv", "single_trans", "single_unknown"]:
        single_run_dir = job_details["run_dir"]
        stream_type_prefix = "STREAM" # Default prefix
        if job_details["type"] == "single_recv":
            stream_type_prefix = "RECV"
        elif job_details["type"] == "single_trans":
            stream_type_prefix = "TRANS"
        
        logger.info(f"[{call_id}] Processing SINGLE ({job_details['type']}) stream from: {single_run_dir.name}")
        
        vocal_stem_path = _find_output_file(single_run_dir, AUDIO_PREPROCESSING_STAGE_KEYWORD, VOCAL_STEM_FILENAME, call_id)
        transcript_path = _find_output_file(single_run_dir, TRANSCRIPTION_STAGE_KEYWORD, TRANSCRIPT_FILENAME, call_id)
        summary_path = _find_output_file(single_run_dir, LLM_SUMMARY_STAGE_KEYWORD, SUMMARY_FILENAME, call_id)

        # Copy soundbites for the single stream
        _copy_soundbite_directories(
            workflow_run_dir=single_run_dir,
            transcription_stage_keyword=TRANSCRIPTION_STAGE_KEYWORD.lower(),
            call_output_dir=call_output_dir,
            stream_prefix=stream_type_prefix, # Use determined prefix
            call_id_for_logging=call_id
        )

        # Copy vocal stem if found (rename to include prefix)
        if vocal_stem_path and vocal_stem_path.exists():
            shutil.copy2(vocal_stem_path, call_output_dir / f"{call_id}_{stream_type_prefix}_{VOCAL_STEM_FILENAME}")
            logger.info(f"[{call_id}] Copied single ({stream_type_prefix}) vocal stem to output directory.")
        if transcript_path:
            shutil.copy2(transcript_path, call_output_dir / f"{call_id}_{stream_type_prefix}_{TRANSCRIPT_FILENAME}")
            logger.info(f"[{call_id}] Copied single ({stream_type_prefix}) transcript to output directory.")
        if summary_path: # Replaces previous direct copy
            shutil.copy2(summary_path, call_output_dir / f"{call_id}_{stream_type_prefix}_{SUMMARY_FILENAME}")
            logger.info(f"[{call_id}] Copied single ({stream_type_prefix}) summary to output directory.")

    # Perform mixing of vocal stems if both are available
    if recv_vocal_stem_path and trans_vocal_stem_path:
        mixed_vocals_output_path = call_output_dir / f"{call_id}_{MIXED_VOCALS_FILENAME}"
        mixing_successful = _mix_stereo_vocals(recv_vocal_stem_path, trans_vocal_stem_path, mixed_vocals_output_path, call_id)
        if mixing_successful:
            logger.info(f"[{call_id}] Vocal mixing completed for pair.")
        else:
            logger.warning(f"[{call_id}] Vocal mixing failed or was skipped for pair due to missing stems or ffmpeg error.")
    else:
        logger.warning(f"[{call_id}] Skipping vocal mixing for pair as one or both vocal stems were not found.")
        
    # Merge transcripts if both are available
    if recv_transcript_path and trans_transcript_path:
        merged_transcript_output_path = call_output_dir / f"{call_id}_{MERGED_TRANSCRIPT_FILENAME}"
        merging_successful = _merge_transcripts(recv_transcript_path, trans_transcript_path, merged_transcript_output_path, call_id)
        if merging_successful:
            logger.info(f"[{call_id}] Transcript merging completed for pair.")
        else:
            logger.warning(f"[{call_id}] Transcript merging failed or was skipped for pair.")
    else:
        logger.warning(f"[{call_id}] Skipping transcript merging for pair as one or both transcript files were not found.")

    # Generate combined summary if merged transcript is available and LLM is configured
    if merged_transcript_output_path and merged_transcript_output_path.exists():
        llm_model_id = config.get('llm_studio_model_identifier') # Ensure this key exists in your processing_config
        
        if llm_model_id:
            llm_call_config = {
                "lm_studio_model_identifier": llm_model_id,
                # llm_module should handle context length internally or via more detailed config
            }
            system_prompt = "Based on the following conversation, write a story about what is going on."
            
            logger.info(f"[{call_id}] Attempting to generate combined summary using LLM.")
            summary_file_path = generate_llm_summary(
                transcript_json_path=merged_transcript_output_path,
                system_prompt=system_prompt,
                llm_config=llm_call_config,
                output_dir=call_output_dir, # The llm_module function should use this
                output_filename=COMBINED_SUMMARY_FILENAME
            )

            if summary_file_path and summary_file_path.exists():
                logger.info(f"[{call_id}] Combined summary generated successfully at \'{summary_file_path}\'.")
            else:
                logger.warning(f"[{call_id}] Combined summary generation failed or was skipped by the LLM module.")
        else:
            logger.warning(f"[{call_id}] Skipping combined summary generation as 'llm_studio_model_identifier' not found in config.")
    elif job_details['type'] == 'pair': # Only if it was a pair type that could have a merged transcript
         logger.warning(f"[{call_id}] Skipping combined summary generation as merged transcript was not available.")

    # TODO: 
    # 1. Files retrieved, mixed, merged. Combined summary attempted.
    # 2. Next: Finalize output structure/saving (mostly done by individual steps). Review overall process.
    
    logger.info(f"Finished processing placeholder for call. Retrieved files copied to output for verification.")


def main():
    parser = argparse.ArgumentParser(description="Process workflow executor outputs to combine paired calls (recv/trans) and handle singles.")
    parser.add_argument("--input_run_dir", type=str, required=True, 
                        help="Path to the base output directory of workflow_executor.py (e.g., workspace/workflow_runs/).")
    parser.add_argument("--output_call_dir", type=str, required=True, 
                        help="Path to the directory where final processed call data will be saved (e.g., workspace/processed_calls/).")
    parser.add_argument("--log_level", type=str, default="INFO", 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        help="Global logging level for the processor (default: INFO).")
    parser.add_argument("--llm_model_id", type=str, default=None, # New argument
                        help="Identifier for the LM Studio model for combined summaries (e.g., 'NousResearch/Hermes-2-Pro-Llama-3-8B'). If not provided, combined summary generation will be skipped.")
    # TODO: Add more arguments later for tones, volume adjustments for final mix, etc.

    args = parser.parse_args()
    setup_logging(args.log_level)

    logger.info(f"Starting Call Processor.")
    logger.info(f"Input Workflow Run Directory: {args.input_run_dir}")
    logger.info(f"Output Processed Calls Directory: {args.output_call_dir}")

    input_run_path = Path(args.input_run_dir)
    output_call_path = Path(args.output_call_dir)

    if not input_run_path.is_dir():
        logger.error(f"Error: Input run directory '{input_run_path}\' does not exist or is not a directory. Exiting.")
        return

    try:
        output_call_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Error: Could not create output directory '{output_call_path}\': {e}. Exiting.")
        return

    # --- Main processing logic ---
    # 1. Find call pairs from the workflow run directory structure
    call_jobs = find_call_pairs(input_run_path)

    if not call_jobs:
        logger.info("No call jobs identified. Exiting.")
        return
        
    logger.info(f"Identified {len(call_jobs)} unique call_ids to process.")

    # 2. For each call_id (job), process it
    # Placeholder for global config that might be passed to process_call
    processing_config = { 
        # Example: 'append_tones': True, 
        #          'mixing_volumes': {'vocals_left': 0.8, 'vocals_right': 0.8},
        #          'llm_studio_model_identifier': 'NousResearch/Hermes-2-Pro-Llama-3-8B' # Example
    }
    if args.llm_model_id:
        processing_config['llm_studio_model_identifier'] = args.llm_model_id
    else:
        # Ensure the key is not present or is None if no CLI arg is given, 
        # to prevent using a hardcoded example if CLI arg is omitted.
        if 'llm_studio_model_identifier' in processing_config:
            logger.info("No --llm_model_id provided. Removing any hardcoded example from processing_config.")
            del processing_config['llm_studio_model_identifier']
            
    successful_calls = 0
    failed_calls = 0

    for call_id, job_details in call_jobs.items():
        try:
            process_call(call_id, job_details, output_call_path, processing_config)
            successful_calls += 1
        except Exception as e:
            logger.error(f"[{call_id}] UNHANDLED EXCEPTION during processing call: {e}", exc_info=True)
            failed_calls +=1
            
    logger.info("--- Call Processing Summary ---")
    logger.info(f"Total call IDs processed: {len(call_jobs)}")
    logger.info(f"Successfully processed (placeholder): {successful_calls}")
    logger.info(f"Failed to process (placeholder): {failed_calls}")
    logger.info(f"Processed call data saved in: {output_call_path}")


if __name__ == "__main__":
    main() 