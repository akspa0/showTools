import os
import re
import logging
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime, timedelta

# Global logger for the compiler
logger = logging.getLogger(__name__)

# Define constants
SHOW_AUDIO_FILENAME = "show_audio.mp3"
SHOW_TIMESTAMPS_FILENAME = "show_timestamps.txt"
SHOW_TRANSCRIPT_FILENAME = "show_transcript.txt"
CALL_TYPE_FILENAME = "call_type.txt"
TRANSCRIPT_FILENAME = "transcript.txt"

# Call type identifiers
CALL_TYPE_PAIR = "pair"
CALL_TYPE_SINGLE_RECV = "single_recv"
CALL_TYPE_SINGLE_TRANS = "single_trans"
CALL_TYPE_SINGLE_UNKNOWN = "single_unknown"


def setup_logging(log_level_str='INFO'):
    """Configures logging for the application."""
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logging.basicConfig(level=log_level, 
                        format='%(asctime)s [%(levelname)s] - %(name)s.%(funcName)s - %(message)s',
                        handlers=[logging.StreamHandler()])
    # Suppress overly verbose logs from libraries if needed
    # logging.getLogger("some_library").setLevel(logging.WARNING)
    logger.info(f"Logging setup with level: {log_level_str}")


def filter_pair_calls(call_folders):
    """
    Filters the call folders to only include those of type 'pair'.
    
    Args:
        call_folders: Dictionary mapping call folder paths to their types.
        
    Returns:
        dict: A filtered dictionary with only pair calls.
    """
    pair_calls = {folder: call_type for folder, call_type in call_folders.items() 
                 if call_type == CALL_TYPE_PAIR}
    
    logger.info(f"Filtered down to {len(pair_calls)} pair calls.")
    return pair_calls


def extract_timestamp_from_folder_name(folder_path: Path):
    """
    Attempts to extract a timestamp from the folder name.
    
    Args:
        folder_path: Path object for the call folder.
        
    Returns:
        str: Extracted timestamp or None if not found.
    """
    # Look for patterns like "call_20250512-020958" embedded in the folder name
    timestamp_match = re.search(r'call_(\d{8}-\d{6})', folder_path.name)
    if timestamp_match:
        return timestamp_match.group(1)
    
    # If embedded timestamp not found, look for a metadata file
    call_metadata_file = folder_path / "call_metadata.json"
    if call_metadata_file.exists():
        try:
            with open(call_metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                if "call_id" in metadata:
                    timestamp_match = re.search(r'call_(\d{8}-\d{6})', metadata["call_id"])
                    if timestamp_match:
                        return timestamp_match.group(1)
        except Exception as e:
            logger.warning(f"Error reading call_metadata.json for {folder_path.name}: {e}")
    
    # If all else fails, use the folder modification time as a rough estimate
    logger.warning(f"Could not extract timestamp from folder name: {folder_path.name}, will use folder mtime")
    return None


def sort_call_dirs_chronologically(pair_calls):
    """
    Sorts the pair call directories chronologically based on timestamps.
    
    Args:
        pair_calls: Dictionary mapping pair call folder paths to their types.
        
    Returns:
        list: Sorted list of call folder paths.
    """
    # Create a list of (folder_path, timestamp_or_none) tuples
    folder_timestamps = []
    for folder_path in pair_calls.keys():
        timestamp = extract_timestamp_from_folder_name(folder_path)
        folder_timestamps.append((folder_path, timestamp))
    
    def timestamp_sort_key(folder_timestamp_tuple):
        """Key function for sorting by timestamp."""
        folder_path, timestamp = folder_timestamp_tuple
        if timestamp:
            # Sort by timestamp first if available
            try:
                # Parse YYYYMMDD-HHMMSS format
                return datetime.strptime(timestamp, "%Y%m%d-%H%M%S")
            except ValueError:
                logger.warning(f"Could not parse timestamp {timestamp} for {folder_path.name}")
        
        # Fallback to folder mtime if timestamp not available or not parseable
        try:
            return datetime.fromtimestamp(folder_path.stat().st_mtime)
        except Exception:
            # Ultimate fallback to folder name string
            return folder_path.name
    
    # Sort by the timestamp or mtime
    sorted_folders_with_timestamps = sorted(folder_timestamps, key=timestamp_sort_key)
    
    # Extract just the folder paths from the sorted list
    sorted_folders = [folder for folder, _ in sorted_folders_with_timestamps]
    
    logger.info(f"Sorted {len(sorted_folders)} pair calls chronologically.")
    return sorted_folders


def get_audio_duration(audio_file_path: Path):
    """
    Gets the duration of an audio file using ffprobe.
    
    Args:
        audio_file_path: Path to the audio file.
        
    Returns:
        float: Duration in seconds or None if unable to determine.
    """
    ffprobe_cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(audio_file_path)
    ]
    
    try:
        process = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=False)
        if process.returncode == 0:
            duration_str = process.stdout.strip()
            try:
                return float(duration_str)
            except ValueError:
                logger.error(f"Could not convert duration string '{duration_str}' to float for {audio_file_path}")
                return None
        else:
            logger.error(f"ffprobe error for {audio_file_path}: {process.stderr}")
            return None
    except FileNotFoundError:
        logger.error("ffprobe command not found. Please ensure ffmpeg is installed and in your PATH.")
        return None
    except Exception as e:
        logger.error(f"Error getting audio duration for {audio_file_path}: {e}")
        return None


def concatenate_mp3_files(sorted_call_folders, output_file: Path):
    """
    Concatenates MP3 files from sorted call folders into a single show audio file.
    
    Args:
        sorted_call_folders: List of call folder paths, sorted chronologically.
        output_file: Path where the concatenated MP3 will be saved.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    # Create a temporary file list for ffmpeg
    temp_file_list = output_file.parent / "show_audio_filelist.txt"
    
    try:
        mp3_files = []
        for folder in sorted_call_folders:
            # Find the MP3 file in the folder - it should be named after the folder
            mp3_file = next((f for f in folder.glob("*.mp3") if f.stem == folder.name), None)
            
            if not mp3_file:
                # Try to find any MP3 file if none matches the folder name
                mp3_files_in_folder = list(folder.glob("*.mp3"))
                if mp3_files_in_folder:
                    mp3_file = mp3_files_in_folder[0]
            
            if mp3_file and mp3_file.exists():
                mp3_files.append(mp3_file)
                logger.debug(f"Added MP3 file for concatenation: {mp3_file}")
            else:
                logger.warning(f"No MP3 file found in folder: {folder}")
        
        if not mp3_files:
            logger.error("No MP3 files found for concatenation.")
            return False
        
        # Write the file list for ffmpeg
        with open(temp_file_list, 'w', encoding='utf-8') as f:
            for mp3_file in mp3_files:
                f.write(f"file '{mp3_file.absolute()}'\n")
        
        # Use ffmpeg to concatenate the files
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output files without asking
            '-f', 'concat',
            '-safe', '0',
            '-i', str(temp_file_list),
            '-c', 'copy',  # Use copy codec for faster processing
            str(output_file)
        ]
        
        logger.debug(f"Executing ffmpeg command for concatenation: {' '.join(ffmpeg_cmd)}")
        process = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)
        
        if process.returncode == 0:
            logger.info(f"Successfully concatenated {len(mp3_files)} MP3 files to {output_file}")
            return True
        else:
            logger.error(f"ffmpeg concatenation failed. Return code: {process.returncode}")
            logger.error(f"ffmpeg stderr: {process.stderr}")
            logger.error(f"ffmpeg stdout: {process.stdout}")
            return False
    
    except Exception as e:
        logger.error(f"Error during MP3 concatenation: {e}")
        return False
    finally:
        # Clean up the temporary file list
        if temp_file_list.exists():
            try:
                os.remove(temp_file_list)
            except Exception as e:
                logger.warning(f"Could not remove temporary file list {temp_file_list}: {e}")


def generate_timestamps_file(sorted_call_folders, show_audio_path: Path, output_file: Path):
    """
    Generates a timestamp file with the start time of each call in the show.
    
    Args:
        sorted_call_folders: List of call folder paths, sorted chronologically.
        show_audio_path: Path to the concatenated show audio file (for verification if needed).
        output_file: Path where the timestamp file will be saved.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        timestamps_lines = []
        current_position = 0.0  # Start position in seconds
        
        for folder in sorted_call_folders:
            # Find the MP3 file in the folder
            mp3_file = next((f for f in folder.glob("*.mp3") if f.stem == folder.name), None)
            
            if not mp3_file:
                # Try to find any MP3 file if none matches the folder name
                mp3_files_in_folder = list(folder.glob("*.mp3"))
                if mp3_files_in_folder:
                    mp3_file = mp3_files_in_folder[0]
            
            if mp3_file and mp3_file.exists():
                # Format the current position as HH:MM:SS
                position_time = str(timedelta(seconds=int(current_position)))
                
                # Get the call name (folder name or from a metadata file if available)
                call_name = folder.name
                
                # Add entry to timestamps file
                timestamps_lines.append(f"{position_time} - {call_name}")
                
                # Get duration for next position
                duration = get_audio_duration(mp3_file)
                if duration:
                    current_position += duration
                else:
                    logger.warning(f"Could not determine duration for {mp3_file}, timestamps may be inaccurate")
                    # Estimate a default duration (5 minutes) if we can't determine it
                    current_position += 300  # 5 minutes in seconds
            else:
                logger.warning(f"No MP3 file found in folder: {folder}, skipping in timestamps")
        
        # Write the timestamps file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Show Timestamps - Format: HH:MM:SS - Call Name\n")
            f.write("# Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("# Total Show Duration: " + str(timedelta(seconds=int(current_position))) + "\n")
            f.write("\n".join(timestamps_lines))
        
        logger.info(f"Generated timestamps file with {len(timestamps_lines)} entries: {output_file}")
        return True
    
    except Exception as e:
        logger.error(f"Error generating timestamps file: {e}")
        return False


def generate_transcript_file(sorted_call_folders, output_file: Path):
    """
    Concatenates transcript files from call folders into a single show transcript.
    
    Args:
        sorted_call_folders: List of call folder paths, sorted chronologically.
        output_file: Path where the concatenated transcript will be saved.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.write("# Combined Show Transcript\n")
            out_f.write("# Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
            
            for folder in sorted_call_folders:
                transcript_file = folder / TRANSCRIPT_FILENAME
                
                if transcript_file.exists():
                    # Write call header
                    out_f.write(f"\n{'=' * 80}\n")
                    out_f.write(f"CALL: {folder.name}\n")
                    out_f.write(f"{'=' * 80}\n\n")
                    
                    # Copy transcript content
                    try:
                        with open(transcript_file, 'r', encoding='utf-8') as in_f:
                            transcript_content = in_f.read()
                        out_f.write(transcript_content)
                        out_f.write("\n\n")
                        logger.debug(f"Added transcript from {folder.name}")
                    except Exception as e:
                        logger.warning(f"Error reading transcript for {folder.name}: {e}")
                        out_f.write(f"[ERROR: Could not read transcript file: {e}]\n\n")
                else:
                    logger.warning(f"No transcript.txt found for {folder.name}")
                    out_f.write(f"\n{'=' * 80}\n")
                    out_f.write(f"CALL: {folder.name}\n")
                    out_f.write(f"{'=' * 80}\n\n")
                    out_f.write("[No transcript available for this call]\n\n")
        
        logger.info(f"Generated combined transcript file: {output_file}")
        return True
    
    except Exception as e:
        logger.error(f"Error generating combined transcript file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Compile processed calls into a 'show' file with audio, timestamps, and transcript.")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Path to the directory containing processed call folders (e.g., final_processed_calls/).")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Path to the directory where the show outputs will be saved (e.g., show_output/).")
    parser.add_argument("--log_level", type=str, default="INFO", 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        help="Global logging level (default: INFO).")
    parser.add_argument("--include_singles", action='store_true',
                        help="Include single calls (not just pairs) in the show compilation.")

    args = parser.parse_args()
    setup_logging(args.log_level)

    logger.info(f"Starting Show Compiler.")
    logger.info(f"Input Directory: {args.input_dir}")
    logger.info(f"Output Directory: {args.output_dir}")

    input_dir_path = Path(args.input_dir)
    output_dir_path = Path(args.output_dir)

    if not input_dir_path.is_dir():
        logger.error(f"Error: Input directory '{input_dir_path}' does not exist or is not a directory. Exiting.")
        return

    try:
        output_dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Error: Could not create output directory '{output_dir_path}': {e}. Exiting.")
        return

    # 1. Scan input directory for call folders
    # [REMOVED: Old scan_final_output_dir and final output directory logic will be replaced by the new final output builder.]
    
    # 2. Filter for pair calls (unless --include_singles is specified)
    if not args.include_singles:
        call_folders = filter_pair_calls(call_folders)
        
        if not call_folders:
            logger.info("No pair calls found. Exiting.")
            return
    
    # 3. Sort call directories chronologically
    sorted_call_folders = sort_call_dirs_chronologically(call_folders)
    
    # 4. Create show audio file by concatenating MP3s
    show_audio_path = output_dir_path / SHOW_AUDIO_FILENAME
    if concatenate_mp3_files(sorted_call_folders, show_audio_path):
        logger.info(f"Show audio file created: {show_audio_path}")
    else:
        logger.error("Failed to create show audio file.")
    
    # 5. Generate timestamps file
    show_timestamps_path = output_dir_path / SHOW_TIMESTAMPS_FILENAME
    if generate_timestamps_file(sorted_call_folders, show_audio_path, show_timestamps_path):
        logger.info(f"Show timestamps file created: {show_timestamps_path}")
    else:
        logger.error("Failed to create show timestamps file.")
    
    # 6. Generate combined transcript file
    show_transcript_path = output_dir_path / SHOW_TRANSCRIPT_FILENAME
    if generate_transcript_file(sorted_call_folders, show_transcript_path):
        logger.info(f"Show transcript file created: {show_transcript_path}")
    else:
        logger.error("Failed to create show transcript file.")
    
    logger.info(f"Show compilation complete. All output files saved to: {output_dir_path}")


if __name__ == "__main__":
    main() 