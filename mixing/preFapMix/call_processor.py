import os
import re
import logging
import argparse
import shutil
import subprocess # Added for ffmpeg
import json # Added for transcript merging
from pathlib import Path
from datetime import datetime # Will be needed for parsing timestamps if not in filenames
from pydub.utils import mediainfo

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
SUGGESTED_NAME_FILENAME_SUFFIX = "_suggested_name.txt" # New
HASHTAGS_FILENAME_SUFFIX = "_hashtags.txt" # New

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
    If the directory contains only stage subfolders (e.g., 00_clap_event_annotation),
    treat the whole directory as a single job for a single audio file.
    """
    logger.info(f"Scanning for call pairs in: {input_run_dir}")
    # Check for stage subfolders (e.g., 00_clap_event_annotation)
    stage_dirs = [d for d in input_run_dir.iterdir() if d.is_dir() and re.match(r"\d{2}_", d.name)]
    if stage_dirs and all(d.is_dir() for d in input_run_dir.iterdir() if d.is_dir()):
        # This is a workflow run for a single audio file
        call_id = input_run_dir.name
        logger.info(f"Detected single audio workflow run. Treating {input_run_dir} as a single job with call_id: {call_id}")
        return {call_id: {"run_dir": input_run_dir, "type": "single_audio"}}
    # Otherwise, use the original logic for call/call-pair folders
    potential_calls = {}
    if not input_run_dir.is_dir():
        logger.error(f"Input run directory not found or not a directory: {input_run_dir}")
        return {}
    for item in input_run_dir.iterdir():
        if item.is_dir():
            dir_name = item.name
            call_id = generate_call_id_from_workflow_dir_name(dir_name)
            if call_id not in potential_calls:
                potential_calls[call_id] = []
            potential_calls[call_id].append(item)
            logger.debug(f"Associated dir '{dir_name}' with call_id '{call_id}'.")
    call_jobs = {}
    for call_id, run_dirs in potential_calls.items():
        recv_dir = None
        trans_dir = None
        for run_dir in run_dirs:
            dir_name = run_dir.name.lower()
            if "recv_out" in dir_name:
                if recv_dir: logger.warning(f"Multiple recv_out dirs found for call_id '{call_id}': {recv_dir.name}, {run_dir.name}. Using first one.")
                else: recv_dir = run_dir
            elif "trans_out" in dir_name:
                if trans_dir: logger.warning(f"Multiple trans_out dirs found for call_id '{call_id}': {trans_dir.name}, {run_dir.name}. Using first one.")
                else: trans_dir = run_dir
        if recv_dir and trans_dir:
            call_jobs[call_id] = {"recv_run_dir": recv_dir, "trans_run_dir": trans_dir, "type": "pair"}
            logger.info(f"Identified PAIR for call_id '{call_id}': recv='{recv_dir.name}', trans='{trans_dir.name}'")
        elif recv_dir:
            call_jobs[call_id] = {"run_dir": recv_dir, "type": "single_recv"}
            logger.info(f"Identified SINGLE_RECV for call_id '{call_id}': '{recv_dir.name}'")
        elif trans_dir:
            call_jobs[call_id] = {"run_dir": trans_dir, "type": "single_trans"}
            logger.info(f"Identified SINGLE_TRANS for call_id '{call_id}': '{trans_dir.name}'")
        else:
            if run_dirs:
                call_jobs[call_id] = {"run_dir": run_dirs[0], "type": "single_unknown"}
                logger.warning(f"Identified SINGLE_UNKNOWN for call_id '{call_id}' from dir '{run_dirs[0].name}'. Contains neither 'recv_out' nor 'trans_out' in name.")
            else:
                logger.error(f"Call ID '{call_id}' had no associated directories after prefix checking. This is unexpected.")
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
            # Use speaker directory name directly without prefixing with stream type
            # unless it's for a known stream type (RECV or TRANS)
            if stream_prefix in ["RECV", "TRANS"]:
                dest_speaker_dir_name = f"{stream_prefix}_{speaker_dir.name}"
            else:
                dest_speaker_dir_name = speaker_dir.name  # Use S0, S1 etc. directly
            
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


def _convert_json_transcript_to_plain_text(json_transcript_path: Path, output_txt_path: Path, call_id_for_logging: str) -> bool:
    try:
        with open(json_transcript_path, 'r', encoding='utf-8') as f_json:
            segments = json.load(f_json) 
        
        plain_text_lines = []
        if isinstance(segments, list):
            for segment in segments:
                speaker = segment.get("speaker", "Unknown Speaker")
                text = segment.get("text", "").strip()
                if text:
                    plain_text_lines.append(f"{speaker}: {text}")
        elif isinstance(segments, dict) and "text" in segments: # Handle non-diarized or simple text transcripts
            plain_text_lines.append(segments["text"])
        else:
            logger.warning(f"[{call_id_for_logging}] JSON transcript at {json_transcript_path} is not a list of segments or a simple text dict. Cannot convert to plain text properly.")
            plain_text_lines.append("Transcript content not in expected format.")

        with open(output_txt_path, 'w', encoding='utf-8') as f_txt:
            f_txt.write("\n".join(plain_text_lines))
        logger.info(f"[{call_id_for_logging}] Converted JSON transcript to plain text: {output_txt_path}")
        return True
    except Exception as e:
        logger.error(f"[{call_id_for_logging}] Failed to convert JSON transcript to plain text: {e}", exc_info=True)
        return False


def _convert_wav_to_mp3(wav_path: Path, mp3_path: Path, call_id_for_logging: str) -> bool:
    """
    Converts a WAV file to MP3 format using ffmpeg.
    
    Args:
        wav_path: Path to the source WAV file
        mp3_path: Path where the MP3 file will be created
        call_id_for_logging: Call ID for logging purposes
        
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    if not wav_path.exists():
        logger.error(f"[{call_id_for_logging}] Source WAV file not found for MP3 conversion: '{wav_path}'")
        return False

    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # Overwrite output files without asking
        '-i', str(wav_path),
        '-codec:a', 'libmp3lame',
        '-qscale:a', '2',  # VBR quality setting (0-9, 0 = best, 9 = worst)
        str(mp3_path)
    ]

    logger.debug(f"[{call_id_for_logging}] Executing ffmpeg command for MP3 conversion: {' '.join(ffmpeg_cmd)}")
    try:
        process = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)
        if process.returncode == 0:
            logger.info(f"[{call_id_for_logging}] Successfully converted WAV to MP3: '{mp3_path}'.")
            return True
        else:
            logger.error(f"[{call_id_for_logging}] ffmpeg MP3 conversion failed for '{mp3_path}'. Return code: {process.returncode}")
            logger.error(f"[{call_id_for_logging}] ffmpeg stderr: {process.stderr}")
            logger.error(f"[{call_id_for_logging}] ffmpeg stdout: {process.stdout}")
            return False
    except FileNotFoundError:
        logger.error(f"[{call_id_for_logging}] ffmpeg command not found. Please ensure ffmpeg is installed and in your PATH.")
        return False
    except Exception as e:
        logger.error(f"[{call_id_for_logging}] An unexpected error occurred during ffmpeg MP3 conversion for '{mp3_path}': {e}")
        return False


def is_valid_audio(file_path, min_duration_sec=5):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        logger.error(f"File does not exist or is empty: {file_path}")
        return False
    try:
        info = mediainfo(file_path)
        duration = float(info.get('duration', 0))
        if duration < min_duration_sec:
            logger.error(f"File too short (<{min_duration_sec}s): {file_path} (duration: {duration:.2f}s)")
            return False
    except Exception as e:
        logger.error(f"Could not get audio info for {file_path}: {e}")
        return False
    return True


def process_call(call_id: str, job_details: dict, output_call_base_dir: Path, config: dict):
    """
    Processes a single call (which could be a pair or a single stream).
    - Creates a dedicated output directory for the call.
    - Copies relevant files (vocals, transcripts, individual summaries).
    - For pairs, mixes audio and merges transcripts.
    - Generates a combined LLM summary, a suggested call name, and hashtag categories for the call.
    """
    logger.info(f"[{call_id}] Starting processing for call.")
    
    # Determine if this is actually a call or just a regular audio file
    is_regular_audio = False
    if job_details["type"].startswith("single_"):
        # Check if this contains any 'call' indicators - if not, treat as regular audio
        input_name = job_details.get("run_dir", Path("unknown")).name.lower()
        if not ("call_" in input_name or "recv_" in input_name or "trans_" in input_name):
            logger.info(f"[{call_id}] This appears to be a regular audio file (not a call): {input_name}")
            is_regular_audio = True
    
    call_output_dir = output_call_base_dir / call_id
    call_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[{call_id}] Output directory: {call_output_dir}")

    recv_vocal_stem_path = None
    trans_vocal_stem_path = None
    recv_transcript_path = None
    trans_transcript_path = None
    # For individual stream summaries and other files
    recv_summary_path = None
    trans_summary_path = None
    
    # Initialize LLM configuration for all three LLM calls
    llm_model_id = config.get("llm_studio_model_identifier")
    if not llm_model_id:
        # Use a default model if not provided
        llm_model_id = "NousResearch/Hermes-2-Pro-Llama-3-8B"
        logger.warning(f"[{call_id}] No LLM model ID provided in config. Using default model: {llm_model_id}")
    llm_config_base = {
        "lm_studio_model_identifier": llm_model_id,
        "lm_studio_base_url": config.get("lm_studio_base_url", "http://localhost:1234/v1"),
        "lm_studio_api_key": config.get("lm_studio_api_key", "lm-studio"),
        "temperature": config.get("temperature", 0.7),
        "max_response_tokens": config.get("max_response_tokens", 1024)
    }


    if job_details["type"] == "pair":
        recv_run_dir = job_details["recv_run_dir"]
        trans_run_dir = job_details["trans_run_dir"]
        logger.info(f"[{call_id}] Processing as PAIR. RECV_DIR: {recv_run_dir.name}, TRANS_DIR: {trans_run_dir.name}")

        # Locate audio_preprocessing stage folders for each leg
        recv_audio_pre_dir = next((d for d in recv_run_dir.iterdir() if d.is_dir() and AUDIO_PREPROCESSING_STAGE_KEYWORD in d.name), None)
        trans_audio_pre_dir = next((d for d in trans_run_dir.iterdir() if d.is_dir() and AUDIO_PREPROCESSING_STAGE_KEYWORD in d.name), None)
        if not recv_audio_pre_dir or not trans_audio_pre_dir:
            logger.warning(f"[{call_id}] Could not find audio_preprocessing stage folder for one or both legs. RECV: {recv_audio_pre_dir}, TRANS: {trans_audio_pre_dir}")
        else:
            # Save/copy stems with explicit channel/leg suffixes
            for leg, audio_pre_dir in [("RECV", recv_audio_pre_dir), ("TRANS", trans_audio_pre_dir)]:
                for stem_type in ["vocals_normalized.wav", "instrumental_normalized.wav"]:
                    src = audio_pre_dir / stem_type
                    if src.exists():
                        out_name = f"{call_id}_{leg}_{stem_type}"
                        dst = audio_pre_dir / out_name
                        shutil.copy2(src, dst)
                        logger.info(f"[{call_id}] Saved/copy stem: {dst}")

        # Always use stems from audio_preprocessing for mixing
        recv_vocal_stem_path = recv_audio_pre_dir / VOCAL_STEM_FILENAME if recv_audio_pre_dir else None
        trans_vocal_stem_path = trans_audio_pre_dir / VOCAL_STEM_FILENAME if trans_audio_pre_dir else None
        logger.info(f"[{call_id}] Using stems for mixing: RECV: {recv_vocal_stem_path}, TRANS: {trans_vocal_stem_path}")

        # Locate and copy/process RECV stream files
        recv_transcript_path = _find_output_file(recv_run_dir, TRANSCRIPTION_STAGE_KEYWORD, TRANSCRIPT_FILENAME, call_id)
        recv_summary_path = _find_output_file(recv_run_dir, LLM_SUMMARY_STAGE_KEYWORD, SUMMARY_FILENAME, call_id)
        if recv_summary_path:
            shutil.copy2(recv_summary_path, call_output_dir / f"{call_id}_recv_stream_summary.txt")
        _copy_soundbite_directories(recv_run_dir, TRANSCRIPTION_STAGE_KEYWORD, call_output_dir, "RECV", call_id)

        # Locate and copy/process TRANS stream files
        trans_transcript_path = _find_output_file(trans_run_dir, TRANSCRIPTION_STAGE_KEYWORD, TRANSCRIPT_FILENAME, call_id)
        trans_summary_path = _find_output_file(trans_run_dir, LLM_SUMMARY_STAGE_KEYWORD, SUMMARY_FILENAME, call_id)
        if trans_summary_path:
            shutil.copy2(trans_summary_path, call_output_dir / f"{call_id}_trans_stream_summary.txt")
        _copy_soundbite_directories(trans_run_dir, TRANSCRIPTION_STAGE_KEYWORD, call_output_dir, "TRANS", call_id)

        # Mix Vocals (only if both are present)
        if recv_vocal_stem_path and trans_vocal_stem_path:
            mixed_vocals_output_path = call_output_dir / f"{call_id}_{MIXED_VOCALS_FILENAME}"
            _mix_stereo_vocals(recv_vocal_stem_path, trans_vocal_stem_path, mixed_vocals_output_path, call_id)
        else:
            logger.warning(f"[{call_id}] Could not mix vocals as one or both vocal stems are missing. RECV: {bool(recv_vocal_stem_path)}, TRANS: {bool(trans_vocal_stem_path)}")

        # Merge Transcripts (only if both are present)
        merged_transcript_path = None
        if recv_transcript_path and trans_transcript_path:
            merged_transcript_output_path = call_output_dir / f"{call_id}_{MERGED_TRANSCRIPT_FILENAME}"
            if _merge_transcripts(recv_transcript_path, trans_transcript_path, merged_transcript_output_path, call_id):
                merged_transcript_path = merged_transcript_output_path
        else:
            logger.warning(f"[{call_id}] Could not merge transcripts as one or both transcript files are missing. RECV: {bool(recv_transcript_path)}, TRANS: {bool(trans_transcript_path)}")
            # Still copy available transcript(s) for downstream use
            if recv_transcript_path:
                shutil.copy2(recv_transcript_path, call_output_dir / f"{call_id}_RECV_{TRANSCRIPT_FILENAME}")
                logger.info(f"[{call_id}] Copied RECV transcript for downstream use.")
            if trans_transcript_path:
                shutil.copy2(trans_transcript_path, call_output_dir / f"{call_id}_TRANS_{TRANSCRIPT_FILENAME}")
                logger.info(f"[{call_id}] Copied TRANS transcript for downstream use.")
            # Set merged_transcript_path to whichever is available for LLM (prefer RECV)
            merged_transcript_path = recv_transcript_path or trans_transcript_path

    elif job_details["type"].startswith("single") or job_details["type"] == "single_audio":
        run_dir = job_details["run_dir"]
        stream_type = job_details["type"].replace("single_", "") if job_details["type"] != "single_audio" else "audio"
        # If this is a regular audio file (not a call), use a generic description
        if is_regular_audio or job_details["type"] == "single_audio":
            stream_type = "audio" # Use a neutral stream type for non-call files
        logger.info(f"[{call_id}] Processing as {job_details['type'].upper()}. DIR: {run_dir.name}")
        audio_pre_dir = next((d for d in run_dir.iterdir() if d.is_dir() and AUDIO_PREPROCESSING_STAGE_KEYWORD in d.name), None)
        if audio_pre_dir:
            for stem_type in ["vocals_normalized.wav", "instrumental_normalized.wav"]:
                src = audio_pre_dir / stem_type
                if src.exists():
                    out_name = f"{call_id}_{stream_type.upper()}_{stem_type}"
                    dst = audio_pre_dir / out_name
                    shutil.copy2(src, dst)
                    logger.info(f"[{call_id}] Saved/copy stem: {dst}")
        # Use transcript as the "merged" transcript for LLM processing
        merged_transcript_path = None
        if audio_pre_dir:
            for stem_type in ["vocals_normalized.wav", "instrumental_normalized.wav"]:
                src = audio_pre_dir / stem_type
                if src.exists():
                    out_name = f"{call_id}_{stream_type.upper()}_{stem_type}"
                    dst = audio_pre_dir / out_name
                    shutil.copy2(src, dst)
                    logger.info(f"[{call_id}] Saved/copy stem: {dst}")
            merged_transcript_path = audio_pre_dir / f"{call_id}_{stream_type.upper()}_vocals_normalized.wav"
            logger.info(f"[{call_id}] Using single stream transcript for LLM: {merged_transcript_path}")
        if audio_pre_dir:
            for stem_type in ["vocals_normalized.wav", "instrumental_normalized.wav"]:
                src = audio_pre_dir / stem_type
                if src.exists():
                    out_name = f"{call_id}_{stream_type.upper()}_{stem_type}"
                    dst = audio_pre_dir / out_name
                    shutil.copy2(src, dst)
                    logger.info(f"[{call_id}] Saved/copy stem: {dst}")
        # For non-call audio, use a more appropriate prefix than "RECV_" or "TRANS_"
        speaker_prefix = "SPEAKER_" if is_regular_audio or job_details["type"] == "single_audio" else stream_type.upper()
        _copy_soundbite_directories(run_dir, TRANSCRIPTION_STAGE_KEYWORD, call_output_dir, speaker_prefix, call_id)
    
    else:
        logger.error(f"[{call_id}] Unknown job type: {job_details['type']}. Skipping.")
        return

    # LLM Processing (Call Name, Synopsis, Hashtags)
    # This section will run if merged_transcript_path is set (either from a pair or a single)
    if merged_transcript_path and merged_transcript_path.exists():
        logger.info(f"[{call_id}] Starting LLM processing using transcript: {merged_transcript_path}")
        logger.debug(f"[{call_id}] LLM configuration: {llm_config_base}")

        # Verify if we have a plain text transcript or need to convert JSON to text
        transcript_for_llm = merged_transcript_path
        if merged_transcript_path.suffix.lower() == '.json':
            # For JSON transcripts, create a plain text version for LLM processing
            plain_text_path = call_output_dir / f"{call_id}_transcript_for_llm.txt"
            logger.info(f"[{call_id}] Converting JSON transcript to plain text for LLM processing: {plain_text_path}")
            if _convert_json_transcript_to_plain_text(merged_transcript_path, plain_text_path, call_id):
                transcript_for_llm = plain_text_path
            else:
                logger.warning(f"[{call_id}] Could not convert JSON transcript to plain text. Using JSON file directly.")

        # Adjust prompts based on whether this is a call or regular audio
        call_name_system_prompt = ""
        call_synopsis_system_prompt = ""
        hashtag_categories_system_prompt = ""
        
        if is_regular_audio or job_details["type"] == "single_audio":
            # For regular audio (non-call), use more generic prompts
            call_name_system_prompt = "Generate a descriptive title for this audio file in under 10 words. Output title only."
            call_synopsis_system_prompt = """Create a concise summary of the audio content covering:
1. Main themes or topics
2. Key points
3. Overall tone or sentiment
Be objective and clear."""
            hashtag_categories_system_prompt = """Provide 3-5 hashtag categories summarizing this audio file's main themes and content.
- Each tag must be a single word or a short_compound_phrase (use underscores).
- Start each tag with '#'.
- Separate tags with spaces.
- Example: #MusicReview #ProductAnnouncement #TechnicalTutorial
- Output only the space-separated hashtags."""
        else:
            # For actual calls, use the original call-specific prompts
            call_name_system_prompt = "Generate a witty, PII-safe call title under 10 words, with no punctuation. Capture conversational absurdity. Output title only."
            call_synopsis_system_prompt = """Create a PII-safe call synopsis:
1. Main reason for call.
2. Key topics.
3. Decisions/outcomes.
4. Action items (and responsible parties, if any).
Objective and clear."""
            hashtag_categories_system_prompt = """Analyze the transcript. Provide a list of 3-5 hashtag categories summarizing the call's main subjects and themes.
- Each tag must be a single word or a short_compound_phrase (use underscores).
- Start each tag with '#'.
- Separate tags with spaces.
- Example: #OrderInquiry #ProductDefect #RefundRequest
- Output only the space-separated hashtags."""

        # 1. Generate Content Name/Title
        call_name_output_filename = f"{call_id}{SUGGESTED_NAME_FILENAME_SUFFIX}"
        logger.info(f"[{call_id}] Generating content name using transcript: {transcript_for_llm}")
        logger.debug(f"[{call_id}] CONTENT NAME - System prompt: {call_name_system_prompt}")
        logger.debug(f"[{call_id}] CONTENT NAME - Output filename: {call_name_output_filename}")
        logger.debug(f"[{call_id}] CONTENT NAME - Output dir: {call_output_dir}")
        
        # Make a deep copy of the config to prevent any chance of cross-talk between requests
        name_llm_config = {key: value for key, value in llm_config_base.items()}
        
        # DIRECT FILE PATH CHECK - Before call name generation
        expected_call_name_path = call_output_dir / call_name_output_filename
        if expected_call_name_path.exists():
            logger.warning(f"[{call_id}] WARNING: Content name file already exists before generation: {expected_call_name_path}")
            
        generated_call_name_path = generate_llm_summary(
            transcript_json_path=transcript_for_llm,  # Use text version when available
            system_prompt=call_name_system_prompt,
            llm_config=name_llm_config,
            output_dir=call_output_dir,
            output_filename=call_name_output_filename
        )
        
        logger.debug(f"[{call_id}] CONTENT NAME - Result path: {generated_call_name_path}")
        
        # DIRECT FILE PATH CHECK - After call name generation
        if expected_call_name_path.exists():
            logger.info(f"[{call_id}] VERIFICATION: Content name file exists at expected path: {expected_call_name_path}")
            try:
                with open(expected_call_name_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    logger.info(f"[{call_id}] Content name file content: '{content}'")
            except Exception as e:
                logger.error(f"[{call_id}] Error reading content name file: {e}")
        else:
            logger.error(f"[{call_id}] CRITICAL ERROR: Content name file was NOT created at expected path: {expected_call_name_path}")
            # List directory contents to see what files were actually created
            try:
                logger.info(f"[{call_id}] Directory contents of {call_output_dir}:")
                for item in call_output_dir.iterdir():
                    logger.info(f"[{call_id}]  - {item.name} ({item.stat().st_size} bytes)")
            except Exception as e:
                logger.error(f"[{call_id}] Error listing directory contents: {e}")
                
        if generated_call_name_path and generated_call_name_path.exists():
            try:
                with open(generated_call_name_path, 'r', encoding='utf-8') as f_name:
                    suggested_call_name = f_name.read().strip()
                logger.info(f"[{call_id}] Generated content name: {suggested_call_name}")
            except Exception as e:
                logger.error(f"[{call_id}] Could not read generated content name from {generated_call_name_path}: {e}")
        else:
            logger.warning(f"[{call_id}] Could not generate or find content name file: {generated_call_name_path}")

        # 2. Generate Content Synopsis
        call_synopsis_output_filename = f"{call_id}_{COMBINED_SUMMARY_FILENAME}"
        logger.info(f"[{call_id}] Generating content synopsis using transcript: {transcript_for_llm}")
        logger.debug(f"[{call_id}] SYNOPSIS - System prompt: {call_synopsis_system_prompt}")
        logger.debug(f"[{call_id}] SYNOPSIS - Output filename: {call_synopsis_output_filename}")
        logger.debug(f"[{call_id}] SYNOPSIS - Output dir: {call_output_dir}")
        
        # Make a deep copy of the config
        synopsis_llm_config = {key: value for key, value in llm_config_base.items()}
        
        # DIRECT FILE PATH CHECK - Before synopsis generation
        expected_synopsis_path = call_output_dir / call_synopsis_output_filename
        if expected_synopsis_path.exists():
            logger.warning(f"[{call_id}] WARNING: Synopsis file already exists before generation: {expected_synopsis_path}")
            
        generated_synopsis_path = generate_llm_summary(
            transcript_json_path=transcript_for_llm,  # Use text version when available
            system_prompt=call_synopsis_system_prompt,
            llm_config=synopsis_llm_config,
            output_dir=call_output_dir,
            output_filename=call_synopsis_output_filename
        )
        
        logger.debug(f"[{call_id}] SYNOPSIS - Result path: {generated_synopsis_path}")
        
        # DIRECT FILE PATH CHECK - After synopsis generation
        if expected_synopsis_path.exists():
            logger.info(f"[{call_id}] VERIFICATION: Synopsis file exists at expected path: {expected_synopsis_path}")
            try:
                with open(expected_synopsis_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()[:100]  # Only read first 100 chars
                    logger.info(f"[{call_id}] Synopsis file content (first 100 chars): '{content}...'")
            except Exception as e:
                logger.error(f"[{call_id}] Error reading synopsis file: {e}")
        else:
            logger.error(f"[{call_id}] CRITICAL ERROR: Synopsis file was NOT created at expected path: {expected_synopsis_path}")
        
        if generated_synopsis_path and generated_synopsis_path.exists():
            logger.info(f"[{call_id}] Content synopsis generated at: {generated_synopsis_path}")
        else:
            logger.warning(f"[{call_id}] Failed to generate content synopsis: {generated_synopsis_path}")

        # 3. Generate Hashtag Categories
        hashtag_output_filename = f"{call_id}{HASHTAGS_FILENAME_SUFFIX}"
        logger.info(f"[{call_id}] Generating hashtag categories using transcript: {transcript_for_llm}")
        logger.debug(f"[{call_id}] HASHTAGS - System prompt: {hashtag_categories_system_prompt}")
        logger.debug(f"[{call_id}] HASHTAGS - Output filename: {hashtag_output_filename}")
        logger.debug(f"[{call_id}] HASHTAGS - Output dir: {call_output_dir}")
        
        # Make a deep copy of the config
        hashtag_llm_config = {key: value for key, value in llm_config_base.items()}
        
        # DIRECT FILE PATH CHECK - Before hashtag generation
        expected_hashtag_path = call_output_dir / hashtag_output_filename
        if expected_hashtag_path.exists():
            logger.warning(f"[{call_id}] WARNING: Hashtag file already exists before generation: {expected_hashtag_path}")
            
        generated_hashtags_path = generate_llm_summary(
            transcript_json_path=transcript_for_llm,  # Use text version when available
            system_prompt=hashtag_categories_system_prompt,
            llm_config=hashtag_llm_config,
            output_dir=call_output_dir,
            output_filename=hashtag_output_filename
        )
        
        logger.debug(f"[{call_id}] HASHTAGS - Result path: {generated_hashtags_path}")
        
        # DIRECT FILE PATH CHECK - After hashtag generation
        if expected_hashtag_path.exists():
            logger.info(f"[{call_id}] VERIFICATION: Hashtag file exists at expected path: {expected_hashtag_path}")
            try:
                with open(expected_hashtag_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    logger.info(f"[{call_id}] Hashtag file content: '{content}'")
            except Exception as e:
                logger.error(f"[{call_id}] Error reading hashtag file: {e}")
        else:
            logger.error(f"[{call_id}] CRITICAL ERROR: Hashtag file was NOT created at expected path: {expected_hashtag_path}")
        
        if generated_hashtags_path and generated_hashtags_path.exists():
            try:
                with open(generated_hashtags_path, 'r', encoding='utf-8') as f_tags:
                    hashtags = f_tags.read().strip()
                logger.info(f"[{call_id}] Generated hashtags: {hashtags}")
            except Exception as e:
                logger.error(f"[{call_id}] Could not read generated hashtags from {generated_hashtags_path}: {e}")
        else:
            logger.warning(f"[{call_id}] Failed to generate hashtag categories: {generated_hashtags_path}")
    
    elif not (merged_transcript_path and merged_transcript_path.exists()):
        logger.warning(f"[{call_id}] Merged transcript not found or not specified. Skipping LLM processing (name, synopsis, hashtags).")

    logger.info(f"[{call_id}] Finished processing call.")


def main():
    parser = argparse.ArgumentParser(description="Process workflow executor outputs to combine paired calls (recv/trans) and handle singles.")
    parser.add_argument("--input_run_dir", type=str, required=True, 
                        help="Path to the base output directory of workflow_executor.py (e.g., workspace/workflow_runs/).")
    parser.add_argument("--output_call_dir", type=str, required=True, 
                        help="Path to the directory where final processed call data will be saved (e.g., workspace/processed_calls/).")
    parser.add_argument("--log_level", type=str, default="INFO", 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        help="Global logging level for the processor (default: INFO).")
    parser.add_argument("--llm_model_id", type=str, default=None, 
                        help="Identifier for the LM Studio model for combined summaries (e.g., 'NousResearch/Hermes-2-Pro-Llama-3-8B'). If not provided, combined summary generation will be skipped.")
    parser.add_argument("--final_output_dir", type=str, default=None,
                        help="Path to the directory where a flatter, more user-friendly version of call data will be saved (optional).")

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
        # Add default configuration here if needed
    }
    if args.llm_model_id:
        # Store the model ID with the key expected by process_call
        processing_config['llm_studio_model_identifier'] = args.llm_model_id
        logger.info(f"Using LLM model: {args.llm_model_id}")
        # Debug: Print the model ID being passed
        logger.debug(f"LLM model ID set in processing_config: {processing_config['llm_studio_model_identifier']}")
    else:
        logger.info("No LLM model specified. LLM processing will be skipped.")
            
    if args.final_output_dir:
        # Create the final output directory and ensure it's a Path object
        final_output_path = Path(args.final_output_dir)
        try:
            logger.info(f"Creating final output directory: {final_output_path}")
            final_output_path.mkdir(parents=True, exist_ok=True)
            processing_config['final_output_dir_path'] = final_output_path
            logger.info(f"Final output (flatter structure) will be saved to: {final_output_path}")
            # Debug: Print the final output directory type and path
            logger.debug(f"Final output directory path type: {type(final_output_path)}")
            logger.debug(f"Final output directory path: {final_output_path}")
        except OSError as e:
            logger.error(f"Error: Could not create final output directory '{final_output_path}': {e}. Final output will be skipped.")
            if 'final_output_dir_path' in processing_config:
                del processing_config['final_output_dir_path']
    
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
    logger.info(f"Successfully processed: {successful_calls}")
    logger.info(f"Failed to process: {failed_calls}")
    logger.info(f"Processed call data saved in: {output_call_path}")
    if 'final_output_dir_path' in processing_config:
        logger.info(f"Final output structure saved in: {processing_config['final_output_dir_path']}")
    else:
        logger.info("No final output structure was generated.")


if __name__ == "__main__":
    main() 