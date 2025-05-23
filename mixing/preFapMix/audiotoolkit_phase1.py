import os
import re
import logging
import argparse
import subprocess
from datetime import datetime
import shutil
import tempfile
import yt_dlp # For URL downloading
from pathlib import Path

import numpy as np
import soundfile as sf
from audiomentations import Compose, LoudnessNormalization, Gain, Limiter

# Configure basic logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
# log = logging.getLogger(__name__)
# Consider making logging more configurable later via a setup function

# --- Constants ---
TARGET_LUFS = -12.0
MIX_SAMPLE_RATE = 44100 # Target sample rate for mixing
TONES_FILE_PATH = "tones.wav" # Assuming it's in the workspace root
DEFAULT_SEPARATOR_MODEL = "mel_band_roformer_vocals_fv4_gabox.ckpt"
DEFAULT_VOCAL_STEM_TARGET_LUFS = -12.0 # User confirmed optimal default
DEFAULT_INSTRUMENTAL_STEM_TARGET_LUFS = -14.0 # User confirmed optimal default
DEFAULT_LIMITER_PEAK_DB = -0.5 # Target peak for vocal stem limiter if enhance_vocals is on
DEFAULT_FINAL_TARGET_LUFS_FFMPEG = -12.0 # Target LUFS for the final mix using ffmpeg loudnorm

TEMP_BASE_DIR = "temp_processing"
FINAL_OUTPUT_DIR_PHASE1 = "phase1_processed_calls"

# --- Helper Functions ---

def setup_logging(log_level_str='INFO'):
    """Configures logging for the application."""
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s [%(levelname)s] - %(funcName)s - %(message)s')
    # Suppress overly verbose logs from libraries if needed
    # logging.getLogger("yt_dlp").setLevel(logging.WARNING)
    # logging.getLogger("subprocess").setLevel(logging.WARNING)

def run_ffmpeg_command(command, description):
    """Run an FFmpeg command with proper logging."""
    logging.info(f"Running FFmpeg for: {description} | CMD: {' '.join(command)}")
    try:
        process = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logging.debug(f"FFmpeg stdout for {description}:")
        logging.debug(process.stdout)
        logging.debug(f"FFmpeg stderr for {description}:")
        logging.debug(process.stderr)
        logging.info(f"{description} completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"{description} failed with error: {e}")
        logging.error(f"FFmpeg stdout (error):")
        logging.error(e.stdout)
        logging.error(f"FFmpeg stderr (error):")
        logging.error(e.stderr)
        return False
    except Exception as ex:
        logging.error(f"Unexpected error during {description}: {ex}")
        return False

def generate_call_id(filepath_or_url, index):
    """Generates a unique call ID from filepath or URL, prioritizing timestamps."""
    filename = os.path.basename(filepath_or_url)
    # Try to extract a timestamp like YYYYMMDD-HHMMSS from a longer timestamp string
    # e.g., from trans_out-01166627677111-725-20250512-020958-1747015798.308.wav
    # we want to get 20250512-020958
    match = re.search(r'(\d{8}-\d{6})-\d{10}(?:\.\d+)?', filename)
    if match:
        return f"call_{match.group(1)}" # Capture only the YYYYMMDD-HHMMSS part
    else:
        # Fallback to sanitized filename part + index for uniqueness
        sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '', Path(filename).stem)
        return f"call_{sanitized_name[:50]}_{index:03d}"

def create_temp_call_directory(base_temp_dir, call_id):
    """Creates a temporary directory for a specific call's processing."""
    call_temp_dir = Path(base_temp_dir) / call_id
    call_temp_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created temporary directory for {call_id}: {call_temp_dir}")
    return call_temp_dir

# --- Core Processing Functions ---

def download_url(url, download_dir, call_id):
    """Downloads audio from a URL using yt-dlp, saving as WAV."""
    logging.info(f"[{call_id}] Downloading URL: {url}")
    Path(download_dir).mkdir(parents=True, exist_ok=True)
    
    # Using call_id directly for the output filename to ensure PII safety and linkage
    output_template = os.path.join(download_dir, f"{call_id}_download.%(ext)s")

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192', # This quality is for the intermediate before WAV conversion
        }],
        'outtmpl': output_template,
        'quiet': False, # Set to True for less verbose download logs
        'noprogress': True,
        'noplaylist': True, # Process only single video if URL is a playlist
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # Expected filename after postprocessing (yt-dlp replaces .%(ext)s with .wav)
            expected_filename = os.path.join(download_dir, f"{call_id}_download.wav")
            
            if os.path.exists(expected_filename):
                logging.info(f"[{call_id}] Successfully downloaded and converted to WAV: {expected_filename}")
                return expected_filename
            else:
                # Fallback: Check if original extension file exists (if WAV conversion somehow failed)
                # This part might need adjustment based on how yt-dlp names files if direct WAV fails
                # This part might need adjustment based on how yt-dlp names files if direct WAV fails
                original_download_path = ydl.prepare_filename(info).replace(info['ext'], 'wav') # Best guess
                if os.path.exists(original_download_path):
                     logging.warning(f"[{call_id}] WAV specific postprocessing might have an issue, but found: {original_download_path}")
                     return original_download_path
                logging.error(f"[{call_id}] Downloaded file not found at expected path: {expected_filename}")
                return None
    except Exception as e:
        logging.error(f"[{call_id}] Error downloading audio from {url}: {e}")
        return None

def identify_channel_pairs_and_singles(file_list, call_id_prefix_map):
    """
    Identifies trans_out/recv_out pairs and single files.
    Returns a list of 'call jobs', where each job is either
    {'type': 'pair', 'id': call_id, 'left_src': path, 'right_src': path} or
    {'type': 'single', 'id': call_id, 'src': path}
    call_id_prefix_map links original filenames to their generated call_ids.
    """
    logging.info("Identifying channel pairs and single files...")
    jobs = []
    paired_files = set()
    
    # Prioritize pairing based on shared call_id derived from timestamp
    files_by_call_id = {}
    for f_path in file_list:
        original_filename = os.path.basename(f_path)
        call_id = call_id_prefix_map.get(original_filename)
        if not call_id:
            logging.warning(f"Could not find call_id for {original_filename}, skipping.")
            continue
        if call_id not in files_by_call_id:
            files_by_call_id[call_id] = []
        files_by_call_id[call_id].append(f_path)

    for call_id, associated_files in files_by_call_id.items():
        left_src = None # This will be recv_out
        right_src = None # This will be trans_out
        
        # Find left and right based on original naming convention within the associated files
        for f_path in associated_files:
            original_filename = os.path.basename(f_path)
            if original_filename.startswith('recv_'): # RECV_OUT is LEFT
                left_src = f_path
            elif original_filename.startswith('trans_'): # TRANS_OUT is RIGHT
                right_src = f_path
        
        if left_src and right_src:
            jobs.append({'type': 'pair', 'id': call_id, 'left_src': left_src, 'right_src': right_src})
            paired_files.add(left_src)
            paired_files.add(right_src)
            logging.info(f"Identified pair for {call_id}: L(recv)='{os.path.basename(left_src)}', R(trans)='{os.path.basename(right_src)}'")
        elif left_src: # Only recv_out found for this ID (becomes a 'left' single)
            jobs.append({'type': 'single', 'id': call_id, 'src': left_src, 'is_left_channel_source': True})
            paired_files.add(left_src)
            logging.info(f"Identified single (recv_ only) for {call_id}: '{os.path.basename(left_src)}'")
        elif right_src: # Only trans_out found for this ID (becomes a 'right' single, though panning might be less relevant for mono)
            jobs.append({'type': 'single', 'id': call_id, 'src': right_src, 'is_left_channel_source': False})
            paired_files.add(right_src)
            logging.info(f"Identified single (trans_ only) for {call_id}: '{os.path.basename(right_src)}'")


    # Process remaining files as singles if they weren't part of a pair from the same call_id
    # This handles cases where a call_id might have been generated but didn't form a pair
    # or for files that got unique call_ids not shared with others.
    for f_path in file_list:
        if f_path not in paired_files:
            call_id = call_id_prefix_map.get(os.path.basename(f_path))
            if call_id: # Ensure it has a call_id
                # Determine if it's notionally a 'left' or 'right' source if it was prefixed
                is_left = os.path.basename(f_path).startswith('recv_') # recv_ is left
                jobs.append({'type': 'single', 'id': call_id, 'src': f_path, 'is_left_channel_source': is_left})
                logging.info(f"Identified single (unpaired) for {call_id}: '{os.path.basename(f_path)}' (is_left_channel_source: {is_left})")
            else:
                logging.warning(f"File {os.path.basename(f_path)} was not paired and had no call_id generated, skipping.")
                
    return jobs


def normalize_source_audio(input_path, output_path, call_id, source_label="", target_lufs=TARGET_LUFS, target_sample_rate=MIX_SAMPLE_RATE):
    """Normalizes a single audio source file to target LUFS and sample rate."""
    logging.info(f"[{call_id}{'-' + source_label if source_label else ''}] Normalizing: {os.path.basename(input_path)} to {target_lufs} LUFS, {target_sample_rate} Hz")
    # FFmpeg's loudnorm filter also handles sample rate conversion if -ar is specified.
    command = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-af", f"loudnorm=I={target_lufs}:TP=-1:LRA=11", # TP=-1 for true peak limiting
        "-ar", str(target_sample_rate),
        str(output_path)
    ]
    desc_suffix = f" {source_label}" if source_label else ""
    if run_ffmpeg_command(command, f"[{call_id}] Normalization for {os.path.basename(input_path)}{desc_suffix}"):
        return str(output_path)
    return None

def separate_audio_stems(normalized_path, stems_output_dir, call_id, source_label="", separator_model_name=DEFAULT_SEPARATOR_MODEL):
    """
    Separates audio into stems using audio-separator CLI.
    Assumes 'audio-separator' is in PATH or can be called directly.
    Returns dict of stem paths: {'vocals': path, 'instrumental': path}
    This is a placeholder for direct library call if preferred/possible later.
    """
    logging.info(f"[{call_id}{'-' + source_label if source_label else ''}] Separating stems for: {os.path.basename(normalized_path)} using model: {separator_model_name}")
    Path(stems_output_dir).mkdir(parents=True, exist_ok=True)

    temp_separator_input_name = "input_for_separator.wav"
    temp_separator_input_path = Path(stems_output_dir) / temp_separator_input_name

    # Copy the normalized audio to this simple, predictable name
    try:
        shutil.copy2(str(normalized_path), str(temp_separator_input_path))
        logging.info(f"[{call_id}] Copied normalized audio to temporary input for separator: {temp_separator_input_path}")
    except Exception as e:
        logging.error(f"[{call_id}{'-' + source_label if source_label else ''}] Failed to copy normalized audio to temp location for separator: {e}")
        return None

    # Command for audio-separator CLI, now acting on the simplified temp name
    cmd = [
        "audio-separator",
        str(temp_separator_input_path), # Process the simply named temp file
        "-m", separator_model_name,
        "--output_dir", str(stems_output_dir),
        "--output_format", "wav",
        "--log_level", "DEBUG"
    ]
    
    logging.info(f"[{call_id}{'-' + source_label if source_label else ''}] Running audio-separator command (on simplified temp input): {' '.join(cmd)}")
    
    stem_paths = {}
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.debug(f"[{call_id}{'-' + source_label if source_label else ''}] audio-separator stdout:\n{process.stdout}")
        logging.debug(f"[{call_id}{'-' + source_label if source_label else ''}] audio-separator stderr:\n{process.stderr}")
        logging.info(f"[{call_id}{'-' + source_label if source_label else ''}] audio-separator completed successfully.")
        
        # Predict output names based on the simple temporary input name
        # Default is usually <input_base>_StemName.wav
        # Observed output: <input_base>_(StemName)_<model_name_without_ckpt_ext>.wav
        model_name_stem = Path(separator_model_name).stem
        separator_output_vocals_name = f"{temp_separator_input_path.stem}_(Vocals)_{model_name_stem}.wav"
        separator_output_instrumental_name = f"{temp_separator_input_path.stem}_(Instrumental)_{model_name_stem}.wav"

        temp_vocals_path = Path(stems_output_dir) / separator_output_vocals_name
        temp_instrumental_path = Path(stems_output_dir) / separator_output_instrumental_name

        # Define our desired final stem names within the stems_output_dir
        final_vocals_name = f"{call_id}_Vocals.wav"
        final_instrumental_name = f"{call_id}_Instrumental.wav"
        final_vocals_path = Path(stems_output_dir) / final_vocals_name
        final_instrumental_path = Path(stems_output_dir) / final_instrumental_name

        if temp_vocals_path.exists():
            os.rename(str(temp_vocals_path), str(final_vocals_path))
            stem_paths['vocals'] = str(final_vocals_path)
            logging.info(f"[{call_id}{'-' + source_label if source_label else ''}] Renamed separated vocals to: {final_vocals_path}")
        else:
            logging.warning(f"[{call_id}{'-' + source_label if source_label else ''}] Separator output vocals stem not found at expected default path: {temp_vocals_path}")
            
        if temp_instrumental_path.exists():
            os.rename(str(temp_instrumental_path), str(final_instrumental_path))
            stem_paths['instrumental'] = str(final_instrumental_path)
            logging.info(f"[{call_id}{'-' + source_label if source_label else ''}] Renamed separated instrumental to: {final_instrumental_path}")
        else:
            logging.warning(f"[{call_id}{'-' + source_label if source_label else ''}] Separator output instrumental stem not found at expected default path: {temp_instrumental_path}")

    except subprocess.CalledProcessError as e:
        logging.error(f"[{call_id}{'-' + source_label if source_label else ''}] audio-separator failed for {os.path.basename(str(temp_separator_input_path))}.")
        logging.error(f"Stdout: {e.stdout}")
        logging.error(f"Stderr: {e.stderr}")
        return None # Ensure we return None so the main loop knows it failed
    except Exception as e:
        logging.error(f"[{call_id}{'-' + source_label if source_label else ''}] Unexpected error during audio separation or renaming: {e}")
        return None # Ensure we return None
    finally:
        # Clean up the temporary input file we created for the separator
        if temp_separator_input_path.exists():
            try:
                os.remove(str(temp_separator_input_path))
                logging.info(f"[{call_id}{'-' + source_label if source_label else ''}] Cleaned up temporary separator input: {temp_separator_input_path}")
            except OSError as e:
                logging.warning(f"[{call_id}{'-' + source_label if source_label else ''}] Could not remove temporary separator input {temp_separator_input_path}: {e}")

    if not stem_paths or 'vocals' not in stem_paths or 'instrumental' not in stem_paths:
        logging.error(f"[{call_id}{'-' + source_label if source_label else ''}] Failed to obtain both vocals and instrumental stems after separation and renaming.")
        return None
        
    return stem_paths


def adjust_stem_volumes(stem_paths_dict, volume_settings, call_id, temp_dir, source_label=""):
    """Adjusts volume of each stem and saves to a new path in temp_dir."""
    logging.info(f"[{call_id}{'-' + source_label if source_label else ''}] Adjusting stem volumes.")
    adjusted_stem_paths = {}
    for stem_name, stem_path in stem_paths_dict.items():
        volume_multiplier = volume_settings.get(stem_name, 1.0) # Default to 1.0 (no change)
        if volume_multiplier == 1.0:
            adjusted_stem_paths[stem_name] = stem_path # No adjustment needed, use original
            logging.debug(f"[{call_id}{'-' + source_label if source_label else ''}] Stem '{stem_name}' volume unchanged (100%).")
            continue

        adjusted_filename = f"{Path(stem_path).stem}_vol_adj.wav"
        output_path = Path(temp_dir) / adjusted_filename
        
        logging.debug(f"[{call_id}{'-' + source_label if source_label else ''}] Adjusting volume for {stem_name} (multiplier: {volume_multiplier}) -> {output_path}")
        command = [
            "ffmpeg", "-y", "-i", str(stem_path),
            "-af", f"volume={volume_multiplier}",
            str(output_path)
        ]
        desc_suffix = f" {source_label}" if source_label else ""
        if run_ffmpeg_command(command, f"[{call_id}] Volume adjustment for {stem_name}{desc_suffix}"):
            adjusted_stem_paths[stem_name] = str(output_path)
        else:
            logging.warning(f"[{call_id}{'-' + source_label if source_label else ''}] Failed to adjust volume for {stem_name}, using original.")
            adjusted_stem_paths[stem_name] = stem_path # Fallback to original if adjustment fails
            
    return adjusted_stem_paths

def sum_stems_to_mono(source_stems_dict, output_mono_path, call_id, source_label=""):
    """
    Sums all provided stems for a source into a single mono track.
    source_stems_dict: {'vocals': path, 'instrumental': path, ...}
    Ensures the output is mono and at the MIX_SAMPLE_RATE.
    """
    logging.info(f"[{call_id}{'-' + source_label if source_label else ''}] Summing stems to mono: {list(source_stems_dict.keys())}")
    
    input_files = list(source_stems_dict.values())
    if not input_files:
        logging.warning(f"[{call_id}{'-' + source_label if source_label else ''}] No stems provided to sum.")
        return None

    ffmpeg_inputs = []
    for f in input_files:
        ffmpeg_inputs.extend(['-i', str(f)])

    # If there's only one input stem, we still process it to ensure it's mono and at target sample rate.
    if len(input_files) == 1:
        filter_complex = f"[0:a]aresample={MIX_SAMPLE_RATE},aformat=channel_layouts=mono[out]"
    else:
        # For multiple inputs, mix them first, then resample and set to mono.
        input_streams = "".join([f"[{i}:a]" for i in range(len(input_files))])
        filter_complex = f"{input_streams}amix=inputs={len(input_files)}:dropout_transition=0,aresample={MIX_SAMPLE_RATE},aformat=channel_layouts=mono[out]"

    command = [
        "ffmpeg", "-y"
    ] + ffmpeg_inputs + [
        "-filter_complex", filter_complex,
        "-map", "[out]", # Map the output of the filter complex
        # "-ac", "1", # aformat=channel_layouts=mono should handle this.
        str(output_mono_path)
    ]
    
    description = f"[{call_id}{'-' + source_label if source_label else ''}] Summing {len(input_files)} stems to mono"
    if run_ffmpeg_command(command, description):
        return str(output_mono_path)
    return None

def mix_stereo_pair(left_mono_path, right_mono_path, output_stereo_path, call_id):
    """Mixes left (recv_out derived) and right (trans_out derived) mono sources into a stereo file with 60/40 panning."""
    logging.info(f"[{call_id}] Mixing stereo pair: L(recv)='{os.path.basename(left_mono_path)}', R(trans)='{os.path.basename(right_mono_path)}'")
    
    # Panning:
    # Input [0:a] (left_mono_path, from recv_out) -> 60% to Output L, 40% to Output R
    # Input [1:a] (right_mono_path, from trans_out) -> 40% to Output L, 60% to Output R
    filter_complex = (
        f"[0:a]pan=stereo|c0=0.6*c0|c1=0.4*c0[L_processed];" \
        f"[1:a]pan=stereo|c0=0.4*c0|c1=0.6*c0[R_processed];" \
        f"[L_processed][R_processed]amix=inputs=2:dropout_transition=0,aresample={MIX_SAMPLE_RATE}[out]"
    )
    
    command = [
        "ffmpeg", "-y",
        "-i", str(left_mono_path),
        "-i", str(right_mono_path),
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-ac", "2", # Ensure stereo output
        str(output_stereo_path)
    ]
    if run_ffmpeg_command(command, f"[{call_id}] Stereo mixing"):
        return str(output_stereo_path)
    return None

def finalize_single_call_audio(source_stems_dict, output_path, call_id):
    """
    Finalizes audio for a single (non-paired) call by summing its stems.
    Output will be mono. If stereo is desired later, it needs different handling.
    """
    logging.info(f"[{call_id}] Finalizing single call audio.")
    return sum_stems_to_mono(source_stems_dict, output_path, call_id, source_label="single")

def append_tones_to_call(call_audio_path, output_final_path, call_id, should_append, tones_file=TONES_FILE_PATH):
    """Appends tones.wav to the call audio if specified."""
    if not should_append:
        logging.info(f"[{call_id}] Skipping tones appending.")
        # If not appending, the input is the final output for this step
        if str(call_audio_path) != str(output_final_path): # Avoid copying if same file
             shutil.copy2(str(call_audio_path), str(output_final_path))
        return str(output_final_path)

    if not Path(tones_file).exists():
        logging.warning(f"[{call_id}] Tones file not found at {tones_file}. Cannot append tones.")
        if str(call_audio_path) != str(output_final_path):
             shutil.copy2(str(call_audio_path), str(output_final_path))
        return str(output_final_path)

    logging.info(f"[{call_id}] Appending tones from {tones_file} to {os.path.basename(call_audio_path)}")
    command = [
        "ffmpeg", "-y",
        "-i", str(call_audio_path),
        "-i", str(tones_file),
        "-filter_complex", f"[0:a][1:a]concat=n=2:v=0:a=1,aresample={MIX_SAMPLE_RATE}[out]", # Resample after concat
        "-map", "[out]",
        str(output_final_path)
    ]
    if run_ffmpeg_command(command, f"[{call_id}] Appending tones"):
        return str(output_final_path)
    else: # Fallback if tone appending fails
        logging.warning(f"[{call_id}] Failed to append tones, using original mixed audio.")
        if str(call_audio_path) != str(output_final_path):
            shutil.copy2(str(call_audio_path), str(output_final_path))
        return str(output_final_path)

def normalize_audio_stem(input_stem_path, output_stem_path, call_id, source_label="", target_lufs=DEFAULT_VOCAL_STEM_TARGET_LUFS, target_sample_rate=MIX_SAMPLE_RATE):
    """Normalizes a single audio stem to a target LUFS, ensuring mono and target sample rate."""
    logging.info(f"[{call_id}{'-' + source_label if source_label else ''}] Normalizing stem: {os.path.basename(input_stem_path)} to {target_lufs} LUFS, mono, {target_sample_rate} Hz")
    
    command = [
        "ffmpeg", "-y", "-i", str(input_stem_path),
        "-af", f"loudnorm=I={target_lufs}:TP=-1:LRA=11,aresample={target_sample_rate},aformat=channel_layouts=mono",
        str(output_stem_path)
    ]
    
    desc_suffix = f" {source_label} stem" if source_label else " stem"
    description = f"[{call_id}] Normalization for {os.path.basename(input_stem_path)}{desc_suffix}"
    
    if run_ffmpeg_command(command, description):
        return str(output_stem_path)
    logging.warning(f"[{call_id}{'-' + source_label if source_label else ''}] Failed to normalize stem {os.path.basename(input_stem_path)}, returning original path.")
    return None # Return None on failure so it can be handled

def normalize_final_mix_ffmpeg(input_path, output_path, call_id, target_lufs, target_sample_rate=MIX_SAMPLE_RATE):
    """Normalizes the final mixed audio to a target LUFS using ffmpeg loudnorm."""
    logging.info(f"[{call_id}] Normalizing final mix with ffmpeg loudnorm: {os.path.basename(input_path)} to {target_lufs} LUFS, TP=-0.5dBFS")
    
    # TP=-0.5 provides a bit of headroom. LRA=11 is a common default.
    # loudnorm is a two-pass filter by default which is good for accuracy.
    command = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-af", f"loudnorm=I={target_lufs}:TP=-0.5:LRA=11",
        "-ar", str(target_sample_rate), # Ensure consistent sample rate
        str(output_path)
    ]
    
    description = f"[{call_id}] Final mix ffmpeg loudnorm for {os.path.basename(input_path)}"
    
    if run_ffmpeg_command(command, description):
        return str(output_path)
    logging.warning(f"[{call_id}] Failed to ffmpeg loudnorm final mix {os.path.basename(input_path)}, returning original path.")
    return None

def enhance_vocal_stem(input_vocal_path, output_vocal_path, call_id, source_label=""):
    """Placeholder for vocal stem enhancement. Currently applies no filters by default.
       If specific enhancements were to be added (e.g., very specific EQ not covered by normalization),
       they would go here, controlled by more specific flags.
    """
    logging.info(f"[{call_id}{'-' + source_label if source_label else ''}] enhance_vocal_stem called for: {os.path.basename(input_vocal_path)}. Applying no filters by default.")
    
    # No filters applied by default in this revised version.
    # If there were actual ffmpeg filters, they would be constructed here.
    # For now, this function will just copy the input to the output 
    # to maintain the file naming pattern and processing chain structure.

    try:
        shutil.copy2(str(input_vocal_path), str(output_vocal_path))
        logging.debug(f"Copied {input_vocal_path} to {output_vocal_path} as no filters in enhance_vocal_stem.")
        return str(output_vocal_path)
    except Exception as e:
        logging.error(f"[{call_id}{'-' + source_label if source_label else ''}] Failed to copy file in enhance_vocal_stem for {os.path.basename(input_vocal_path)}: {e}")
        return None

# --- Audiomentations-based processing functions ---

def normalize_stem_audiomentations(input_path, output_path, target_lufs, call_id, source_label=""):
    """Normalizes a stem to target_lufs using audiomentations.LoudnessNormalization."""
    logging.info(f"[{call_id}{'-' + source_label if source_label else ''}] Normalizing stem (audiomentations): {os.path.basename(input_path)} to {target_lufs} LUFS")
    try:
        samples, sample_rate = sf.read(input_path, dtype='float32')
        
        # Ensure mono for LoudnessNormalization as it expects single channel or applies to first if multi
        # A more robust mono conversion might be needed if sources are surprisingly multi-channel post-separation
        if samples.ndim > 1 and samples.shape[1] > 1:
            samples = np.mean(samples, axis=1) 
            logging.debug(f"Mixed stem {input_path} to mono for normalization.")

        augment = Compose([
            LoudnessNormalization(min_lufs_in_db=float(target_lufs), max_lufs_in_db=float(target_lufs), p=1.0)
        ])
        processed_samples = augment(samples=samples, sample_rate=sample_rate)
        
        # LoudnessNormalization does not change sample rate. We assume output from separator is MIX_SAMPLE_RATE.
        # If not, resampling would be needed here or before this stage.
        sf.write(output_path, processed_samples, sample_rate)
        return str(output_path)
    except Exception as e:
        logging.error(f"[{call_id}{'-' + source_label if source_label else ''}] Error during audiomentations LoudnessNormalization for {input_path}: {e}")
        return None

def adjust_volume_audiomentations(input_path, output_path, volume_multiplier, call_id, source_label=""):
    """Adjusts volume of a stem using audiomentations.Gain."""
    if volume_multiplier == 0:
        logging.info(f"[{call_id}{'-' + source_label if source_label else ''}] Applying 0x volume (silence) to: {os.path.basename(input_path)}")
        try:
            _ , samplerate = sf.read(input_path) # Read to get samplerate and info
            info = sf.info(input_path)
            duration_frames = int(info.duration * samplerate)
            num_channels = info.channels
            # Create silent samples based on original file's properties
            silent_samples = np.zeros((duration_frames, num_channels) if num_channels > 1 else duration_frames, dtype='float32')
            sf.write(output_path, silent_samples, samplerate)
            return str(output_path)
        except Exception as e:
            logging.error(f"[{call_id}{'-' + source_label if source_label else ''}] Error creating silence for {input_path}: {e}")
            return None

    gain_db = 20 * np.log10(volume_multiplier)
    logging.info(f"[{call_id}{'-' + source_label if source_label else ''}] Adjusting volume by {volume_multiplier}x ({gain_db:.2f} dB) (audiomentations): {os.path.basename(input_path)}")
    try:
        samples, sample_rate = sf.read(input_path, dtype='float32')
        augment = Compose([
            Gain(min_gain_in_db=gain_db, max_gain_in_db=gain_db, p=1.0)
        ])
        processed_samples = augment(samples=samples, sample_rate=sample_rate)
        sf.write(output_path, processed_samples, sample_rate)
        return str(output_path)
    except Exception as e:
        logging.error(f"[{call_id}{'-' + source_label if source_label else ''}] Error during audiomentations Gain for {input_path}: {e}")
        return None

def limit_stem_audiomentations(input_path, output_path, peak_db, call_id, source_label=""):
    """Applies a peak limiter to a stem using audiomentations.Limiter."""
    logging.info(f"[{call_id}{'-' + source_label if source_label else ''}] Applying limiter (audiomentations) to {peak_db} dBFS for: {os.path.basename(input_path)}")
    try:
        samples, sample_rate = sf.read(input_path, dtype='float32')
        augment = Compose([
            # Using fixed short attack/release. Limiter affects threshold and above.
            Limiter(min_threshold_db=float(peak_db), max_threshold_db=float(peak_db), min_attack=0.001, max_attack=0.002, min_release=0.05, max_release=0.1, p=1.0)
        ])
        processed_samples = augment(samples=samples, sample_rate=sample_rate)
        sf.write(output_path, processed_samples, sample_rate)
        return str(output_path)
    except Exception as e:
        logging.error(f"[{call_id}{'-' + source_label if source_label else ''}] Error during audiomentations Limiter for {input_path}: {e}")
        return None

# --- Main Orchestration ---

def process_calls_phase1(input_sources: list, base_output_dir: str, config: dict):
    """
    Main orchestrator for Phase 1 processing.
    input_sources: List of file paths or URLs.
    base_output_dir: Root directory for all outputs.
    config: Dictionary holding settings like volume, model choices, tones options.
    """
    # setup_logging is now called from if __name__ == "__main__" block
    # logging.info(f"Starting Phase 1 audio processing. Base output: {base_output_dir}")

    run_timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    current_run_output_dir = Path(base_output_dir) / run_timestamp

    # Ensure base output directories exist within the timestamped run folder
    phase1_output_path = current_run_output_dir / FINAL_OUTPUT_DIR_PHASE1
    temp_processing_path = current_run_output_dir / TEMP_BASE_DIR
    
    try:
        phase1_output_path.mkdir(parents=True, exist_ok=True)
        temp_processing_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Starting Phase 1 audio processing. Output for this run: {current_run_output_dir}")
    except OSError as e:
        logging.error(f"Failed to create output directories for run {run_timestamp} in {base_output_dir}: {e}")
        return # Cannot proceed if output dirs can't be made

    download_dir = temp_processing_path / "downloads" # Centralized download location
    download_dir.mkdir(parents=True, exist_ok=True)

    local_files = []
    call_id_map = {} # Maps original basename to generated call_id for pairing

    # 1. Download URLs and generate initial call_ids for all inputs
    for i, source_item in enumerate(input_sources):
        original_basename = os.path.basename(source_item)
        call_id = generate_call_id(original_basename, i) # Generate call_id based on original name
        call_id_map[original_basename] = call_id # Store it

        if source_item.startswith(('http://', 'https://')):
            downloaded_file = download_url(source_item, download_dir, call_id)
            if downloaded_file:
                local_files.append(downloaded_file)
                # Update map if downloaded filename differs (it shouldn't with current download_url)
                # call_id_map[os.path.basename(downloaded_file)] = call_id 
            else:
                logging.error(f"Failed to download {source_item}, skipping.")
        elif Path(source_item).is_file():
            local_files.append(source_item)
        else:
            logging.warning(f"Input item '{source_item}' is not a valid file or URL, skipping.")

    if not local_files:
        logging.info("No valid local files to process after download step.")
        return

    # 2. Identify pairs and singles from the now localized file list
    call_jobs = identify_channel_pairs_and_singles(local_files, call_id_map)
    
    separator_model = config.get('separator_model', DEFAULT_SEPARATOR_MODEL)
    volume_settings = config.get('volume_settings', {'vocals': 1.0, 'instrumental': 0.75})
    append_tones_singles = config.get('append_tones_for_singles', True)
    append_tones_pairs = config.get('append_tones_for_pairs', True)
    enhance_vocals_flag = config.get('enhance_vocals', True)
    vocal_stem_target_lufs = config.get('vocal_stem_target_lufs', DEFAULT_VOCAL_STEM_TARGET_LUFS)
    instrumental_stem_target_lufs = config.get('instrumental_stem_target_lufs', DEFAULT_INSTRUMENTAL_STEM_TARGET_LUFS)
    limiter_peak_db = config.get('limiter_peak_db', DEFAULT_LIMITER_PEAK_DB)
    save_intermediate_stems_flag = config.get('save_intermediate_stems', False)

    # Config for audiomentations-based final processing
    cfg_final_target_lufs_ffmpeg = config.get('final_target_lufs_ffmpeg', DEFAULT_FINAL_TARGET_LUFS_FFMPEG)

    # 3. Process each call job
    for job in call_jobs:
        call_id = job['id']
        logging.info(f"Processing job for Call ID: {call_id}, Type: {job['type']}")
        call_temp_dir = create_temp_call_directory(temp_processing_path, call_id)

        processed_sources_for_mixing = {} # To hold paths to mono content for L and R

        if job['type'] == 'pair':
            sources_to_process = [
                {'label': 'left', 'path': job['left_src']},
                {'label': 'right', 'path': job['right_src']}
            ]
        else: # single
            sources_to_process = [{'label': 'single', 'path': job['src']}]

        all_sources_processed_successfully = True
        for source_info in sources_to_process:
            source_label = source_info['label'] # 'left', 'right', or 'single'
            original_source_path = source_info['path']
            
            # Current input for separation is the original (or downloaded) source path
            input_for_separation = original_source_path
            
            # a. Separate stems (using input_for_separation)
            stems_output_dir = call_temp_dir / f"{source_label}_stems"
            separated_stems = separate_audio_stems(input_for_separation, stems_output_dir, call_id, source_label, separator_model)
            if not separated_stems:
                 all_sources_processed_successfully = False; break
            
            if save_intermediate_stems_flag:
                if 'vocals' in separated_stems and Path(separated_stems['vocals']).exists():
                    shutil.copy2(separated_stems['vocals'], call_temp_dir / f"01_{call_id}_{source_label}_vocals_separated.wav")
                if 'instrumental' in separated_stems and Path(separated_stems['instrumental']).exists():
                    shutil.copy2(separated_stems['instrumental'], call_temp_dir / f"01_{call_id}_{source_label}_instrumental_separated.wav")

            # --- Start Audiomentations Processing ---
            processed_stems_for_summing = {}

            # Process Vocal Stem
            if 'vocals' in separated_stems and Path(separated_stems['vocals']).exists():
                current_vocal_path = separated_stems['vocals']
                
                # 1. Loudness Normalization (Audiomentations)
                am_norm_vocal_filename = f"{call_id}_{source_label}_vocals_am_norm.wav"
                am_norm_vocal_path = normalize_stem_audiomentations(
                    current_vocal_path,
                    call_temp_dir / am_norm_vocal_filename,
                    target_lufs=vocal_stem_target_lufs,
                    call_id=call_id,
                    source_label=f"{source_label}_vocals"
                )
                if am_norm_vocal_path:
                    current_vocal_path = am_norm_vocal_path
                    if save_intermediate_stems_flag:
                        shutil.copy2(current_vocal_path, call_temp_dir / f"02_{call_id}_{source_label}_vocals_am_normalized.wav")
                else:
                    logging.warning(f"[{call_id}-{source_label}] Audiomentations vocal normalization failed. Using previous vocal stem.")
                
                # 2. Optional Limiter (Audiomentations) - Controlled by enhance_vocals_flag
                if enhance_vocals_flag: # This flag now controls only this limiter for vocals
                    logging.info(f"[{call_id}-{source_label}] Applying audiomentations limiter to vocal stem.")
                    am_limited_vocal_filename = f"{call_id}_{source_label}_vocals_am_limited.wav"
                    am_limited_vocal_path = limit_stem_audiomentations(
                        current_vocal_path,
                        call_temp_dir / am_limited_vocal_filename,
                        peak_db=limiter_peak_db,
                        call_id=call_id,
                        source_label=f"{source_label}_vocals"
                    )
                    if am_limited_vocal_path:
                        current_vocal_path = am_limited_vocal_path
                        if save_intermediate_stems_flag:
                             shutil.copy2(current_vocal_path, call_temp_dir / f"03_{call_id}_{source_label}_vocals_am_limited.wav")
                    else:
                        logging.warning(f"[{call_id}-{source_label}] Audiomentations vocal limiting failed. Using un-limited vocal stem.")
                
                # 3. Volume Adjustment (Audiomentations)
                am_vol_adj_vocal_filename = f"{call_id}_{source_label}_vocals_am_vol_adj.wav"
                am_vol_adj_vocal_path = adjust_volume_audiomentations(
                    current_vocal_path,
                    call_temp_dir / am_vol_adj_vocal_filename,
                    volume_multiplier=volume_settings.get('vocals', 1.0),
                    call_id=call_id,
                    source_label=f"{source_label}_vocals"
                )
                if am_vol_adj_vocal_path:
                    processed_stems_for_summing['vocals'] = am_vol_adj_vocal_path
                    if save_intermediate_stems_flag:
                        shutil.copy2(am_vol_adj_vocal_path, call_temp_dir / f"04_{call_id}_{source_label}_vocals_am_vol_adjusted.wav")
                else:
                    logging.warning(f"[{call_id}-{source_label}] Audiomentations vocal volume adjustment failed. Using previous vocal stem for summing.")
                    processed_stems_for_summing['vocals'] = current_vocal_path # Fallback
            else:
                logging.error(f"[{call_id}-{source_label}] Vocal stem missing after separation. Cannot process vocals.")
                all_sources_processed_successfully = False; break


            # Process Instrumental Stem
            if 'instrumental' in separated_stems and Path(separated_stems['instrumental']).exists():
                current_instrumental_path = separated_stems['instrumental']

                # 1. Loudness Normalization (Audiomentations) - REINTRODUCED for instrumentals
                am_norm_instrumental_filename = f"{call_id}_{source_label}_instrumental_am_norm.wav"
                am_norm_instrumental_path = normalize_stem_audiomentations(
                    current_instrumental_path,
                    call_temp_dir / am_norm_instrumental_filename,
                    target_lufs=instrumental_stem_target_lufs, # Use the dedicated LUFS target
                    call_id=call_id,
                    source_label=f"{source_label}_instrumental"
                )
                if am_norm_instrumental_path:
                    current_instrumental_path = am_norm_instrumental_path
                    if save_intermediate_stems_flag:
                        shutil.copy2(current_instrumental_path, call_temp_dir / f"02_{call_id}_{source_label}_instrumental_am_normalized.wav")
                else:
                    logging.warning(f"[{call_id}-{source_label}] Audiomentations instrumental normalization failed. Using previous instrumental stem.")

                # 2. Volume Adjustment (Audiomentations) - Applied to normalized instrumental
                am_vol_adj_instrumental_filename = f"{call_id}_{source_label}_instrumental_am_vol_adj.wav"
                am_vol_adj_instrumental_path = adjust_volume_audiomentations(
                    current_instrumental_path, 
                    call_temp_dir / am_vol_adj_instrumental_filename,
                    volume_multiplier=volume_settings.get('instrumental', 1.0),
                    call_id=call_id,
                    source_label=f"{source_label}_instrumental"
                )
                if am_vol_adj_instrumental_path:
                    processed_stems_for_summing['instrumental'] = am_vol_adj_instrumental_path
                    if save_intermediate_stems_flag:
                        # Consistent numbering: 01_sep, 02_norm_instr, 03_vol_adj_instr (if vocal has norm+limit+vol)
                        # Let's use a distinct step number for instrumental vol adj after norm.
                        shutil.copy2(am_vol_adj_instrumental_path, call_temp_dir / f"03_{call_id}_{source_label}_instrumental_am_vol_adjusted.wav")
                else:
                    logging.warning(f"[{call_id}-{source_label}] Audiomentations instrumental volume adjustment failed. Using normalized instrumental stem for summing.")
                    processed_stems_for_summing['instrumental'] = current_instrumental_path # Fallback to normalized instrumental stem
            else:
                logging.error(f"[{call_id}-{source_label}] Instrumental stem missing after separation. Cannot process instrumentals.")
                all_sources_processed_successfully = False; break
            
            if not all_sources_processed_successfully: break # Break from processing sources for this job
            
            # Check if we have the necessary stems after audiomentations processing
            if 'vocals' not in processed_stems_for_summing or 'instrumental' not in processed_stems_for_summing:
                logging.error(f"[{call_id}-{source_label}] Missing processed 'vocals' or 'instrumental' stem after audiomentations, cannot proceed with this source.")
                all_sources_processed_successfully = False; break

            # --- End Audiomentations Processing ---

            # f. Sum stems for this source to a single mono track (using audiomentations processed stems)
            summed_mono_filename = f"{call_id}_{source_label}_summed_mono.wav"
            summed_mono_path = sum_stems_to_mono(processed_stems_for_summing, call_temp_dir / summed_mono_filename, call_id, source_label)
            if not summed_mono_path:
                all_sources_processed_successfully = False; break
            
            if save_intermediate_stems_flag and summed_mono_path and Path(summed_mono_path).exists():
                shutil.copy2(summed_mono_path, call_temp_dir / f"05_{call_id}_{source_label}_summed_mono_final.wav")
            
            processed_sources_for_mixing[source_label] = summed_mono_path
        
        if not all_sources_processed_successfully:
            logging.error(f"[{call_id}] Processing failed for one or more sources, skipping final mix for this call.")
            continue

        # g. Mix (if pair) or finalize (if single)
        mixed_call_audio_path = None
        if job['type'] == 'pair':
            if 'left' in processed_sources_for_mixing and 'right' in processed_sources_for_mixing:
                mixed_stereo_filename = f"{call_id}_mixed_stereo.wav"
                mixed_call_audio_path = mix_stereo_pair(
                    processed_sources_for_mixing['left'],
                    processed_sources_for_mixing['right'],
                    call_temp_dir / mixed_stereo_filename,
                    call_id
                )
            else:
                logging.error(f"[{call_id}] Missing processed left or right source for pair, cannot mix.")
                continue
        else: # single
             # For single, the "mixed" audio is just its summed mono track
            mixed_call_audio_path = processed_sources_for_mixing.get('single')
            if not mixed_call_audio_path:
                logging.error(f"[{call_id}] Missing processed audio for single call, cannot finalize.")
                continue


        if not mixed_call_audio_path:
            logging.error(f"[{call_id}] Failed to produce mixed/finalized audio for the call.")
            continue
            
        # FINAL MIX PROCESSING (FFMPEG LOUDNORM)
        processed_for_tones_input_path = mixed_call_audio_path # Default if processing fails

        ffmpeg_norm_final_filename = f"{call_id}_ffmpeg_final_norm.wav"
        ffmpeg_norm_final_path = normalize_final_mix_ffmpeg(
            mixed_call_audio_path, 
            call_temp_dir / ffmpeg_norm_final_filename,
            call_id,
            target_lufs=cfg_final_target_lufs_ffmpeg
        )

        if ffmpeg_norm_final_path:
            processed_for_tones_input_path = ffmpeg_norm_final_path
            if save_intermediate_stems_flag:
                # This is now the main final processing step before tones (effectively step 06)
                shutil.copy2(processed_for_tones_input_path, call_temp_dir / f"06_{call_id}_ffmpeg_final_normalized.wav")
        else:
            logging.warning(f"[{call_id}] FFMPEG final mix normalization failed. Proceeding with un-normalized mix for tone appending.")
            # processed_for_tones_input_path remains mixed_call_audio_path
            
        # h. Append tones
        final_output_filename = f"{call_id}_final.wav"
        final_call_path = phase1_output_path / final_output_filename
        
        should_append = append_tones_pairs if job['type'] == 'pair' else append_tones_singles
        
        append_tones_to_call(processed_for_tones_input_path, final_call_path, call_id, should_append)
        logging.info(f"[{call_id}] Final processed call saved to: {final_call_path}")

        # i. Cleanup temp call_id directory (optional, can be done at the very end)
        # shutil.rmtree(call_temp_dir) 
        # logging.info(f"[{call_id}] Cleaned up temporary directory: {call_temp_dir}")

    logging.info("Phase 1 processing finished.")
    # Optional: Cleanup the main temp_processing_path/downloads if desired
    # shutil.rmtree(download_dir)


# --- Argparse and main() ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 Audio Processing Toolkit: Normalization, Separation, Mixing, Tones.")
    parser.add_argument("input_sources", nargs='*', help="List of input audio file paths or URLs (optional if --input_dir is used).")
    parser.add_argument("--input_dir", help="Directory containing input audio files to process (primarily trans_ and recv_ prefixed files).")
    parser.add_argument("--output_dir", required=True, help="Base directory where output will be saved (e.g., ./output).")
    
    parser.add_argument("--separator_model", default=DEFAULT_SEPARATOR_MODEL, help=f"Name of the audio separator model (default: {DEFAULT_SEPARATOR_MODEL}).")
    parser.add_argument("--volume_vocals", type=float, default=1.0, help="Volume multiplier for vocals stem (default: 1.0 for 100%).")
    parser.add_argument("--volume_instrumental", type=float, default=0.75, help="Volume multiplier for instrumental stem (default: 0.75 for 75%).")
    # Add more volumes if other stems become common, e.g., --volume_bass, --volume_drums
    parser.add_argument("--vocal_stem_target_lufs", type=float, default=DEFAULT_VOCAL_STEM_TARGET_LUFS, help=f"Target LUFS for isolated vocal stem normalization (default: {DEFAULT_VOCAL_STEM_TARGET_LUFS}).")
    parser.add_argument("--instrumental_stem_target_lufs", type=float, default=DEFAULT_INSTRUMENTAL_STEM_TARGET_LUFS, help=f"Target LUFS for isolated instrumental stem normalization (default: {DEFAULT_INSTRUMENTAL_STEM_TARGET_LUFS}).")
    parser.add_argument("--limiter_peak_db", type=float, default=DEFAULT_LIMITER_PEAK_DB, help=f"Target peak dBFS for vocal stem limiter (default: {DEFAULT_LIMITER_PEAK_DB}). Used if --enhance_vocals is enabled.")
    parser.add_argument("--enhance_vocals", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable vocal stem enhancement (limiter stage) (default: enabled).")
    parser.add_argument("--save_intermediate_stems", action='store_true', help="Save all intermediate stems for debugging loudness levels.")
    parser.add_argument("--final_target_lufs_ffmpeg", type=float, default=DEFAULT_FINAL_TARGET_LUFS_FFMPEG, help=f"Target LUFS for the final mix using ffmpeg loudnorm (default: {DEFAULT_FINAL_TARGET_LUFS_FFMPEG}).")

    parser.add_argument("--no_tones_singles", action="store_false", dest="append_tones_for_singles", help="Disable appending tones for single (non-paired) calls.")
    parser.add_argument("--no_tones_pairs", action="store_false", dest="append_tones_for_pairs", help="Disable appending tones for paired calls.")
    parser.set_defaults(append_tones_for_singles=True, append_tones_for_pairs=True)

    parser.add_argument("--log_level", default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging level.")

    args = parser.parse_args()
    
    # Setup logging as early as possible
    setup_logging(args.log_level)

    all_items_to_process = []
    supported_audio_extensions = ['.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg'] # Define supported extensions

    if args.input_dir:
        if not Path(args.input_dir).is_dir():
            logging.error(f"Specified input directory does not exist or is not a directory: {args.input_dir}")
            # Consider exiting if only input_dir was meant to be used and it's invalid
        else:
            logging.info(f"Scanning directory '{args.input_dir}' for 'trans_' and 'recv_' prefixed audio files...")
            for entry_name in os.listdir(args.input_dir):
                full_path = Path(args.input_dir) / entry_name
                if full_path.is_file():
                    filename_lower = entry_name.lower()
                    if filename_lower.startswith(('trans_', 'recv_')) and full_path.suffix.lower() in supported_audio_extensions:
                        logging.debug(f"Found matching file in directory: {full_path}")
                        all_items_to_process.append(str(full_path))
                    elif full_path.suffix.lower() in supported_audio_extensions:
                        logging.debug(f"Found audio file in directory (will be ignored unless also in input_sources): {full_path}")
                else:
                    logging.debug(f"Skipping non-file entry in input directory: {entry_name}")
    
    # Add explicitly provided files/URLs
    if args.input_sources:
        for source_item in args.input_sources:
            if source_item.startswith(('http://', 'https://')):
                logging.debug(f"Adding URL source: {source_item}")
                all_items_to_process.append(source_item)
            else:
                # Check if it's a file and exists
                if Path(source_item).is_file():
                    if Path(source_item).suffix.lower() in supported_audio_extensions:
                        logging.debug(f"Adding explicit file source: {source_item}")
                        all_items_to_process.append(source_item)
                    else:
                        logging.warning(f"Explicitly provided file '{source_item}' is not a supported audio type, skipping.")
                else:
                    # Check if it might be a glob pattern or a non-audio file from a directory scan if logic was different
                    # For now, strict file existence for non-URL explicit inputs
                    logging.warning(f"Explicitly provided file '{source_item}' does not exist or is not a file, skipping.")

    # Remove duplicates if a file was in input_dir and also explicitly listed
    all_items_to_process = sorted(list(set(all_items_to_process)))

    if not all_items_to_process:
        logging.info("No input files or URLs to process. Exiting.")
        # exit() # Or return if this was in a main function

    config_dict = {
        'separator_model': args.separator_model,
        'volume_settings': {
            'vocals': args.volume_vocals,
            'instrumental': args.volume_instrumental,
            # 'bass': args.volume_bass, # Example for future
        },
        'append_tones_for_singles': args.append_tones_for_singles,
        'append_tones_for_pairs': args.append_tones_for_pairs,
        'enhance_vocals': args.enhance_vocals,
        'vocal_stem_target_lufs': args.vocal_stem_target_lufs,
        'instrumental_stem_target_lufs': args.instrumental_stem_target_lufs,
        'save_intermediate_stems': args.save_intermediate_stems,
        'limiter_peak_db': args.limiter_peak_db,
        'final_target_lufs_ffmpeg': args.final_target_lufs_ffmpeg,
        'log_level': args.log_level,
    }

    # Call process_calls_phase1 with the combined and filtered list
    if all_items_to_process: # Ensure list is not empty before processing
        process_calls_phase1(all_items_to_process, args.output_dir, config_dict)
    else:
        logging.info("No valid audio sources found to process after filtering. Check your inputs and --input_dir.") 