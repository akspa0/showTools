import os
import re
import logging
import argparse
import subprocess
from datetime import datetime
import shutil
import gc
import json
import tempfile
from pydub.utils import mediainfo

logging.basicConfig(level=logging.INFO)

def run_ffmpeg_command(command, description):
    """Run an FFmpeg command with proper logging."""
    try:
        process = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logging.info(f"{description} completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"{description} failed with error: {e}")
        logging.error(f"Error output: {e.stderr}")
        return False
    except Exception as ex:
        logging.error(f"Unexpected error during {description}: {ex}")
        return False

def get_audio_info(file_path):
    """Get audio file information using FFmpeg."""
    command = [
        "ffmpeg", "-i", file_path, 
        "-hide_banner", "-f", "null", "-"
    ]
    
    audio_info = {'sample_rate': None, 'channels': None, 'duration': None}
    
    try:
        result = subprocess.run(command, stderr=subprocess.PIPE, check=False, text=True)
        stderr = result.stderr
        
        # Extract sample rate
        sample_rate_match = re.search(r'(\d+) Hz', stderr)
        if sample_rate_match:
            audio_info['sample_rate'] = int(sample_rate_match.group(1))
        
        # Extract channel count
        channel_match = re.search(r'Audio: .+?, (\d+) channels', stderr)
        if channel_match:
            audio_info['channels'] = int(channel_match.group(1))
            
        # Extract duration
        duration_match = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)', stderr)
        if duration_match:
            hours, minutes, seconds = duration_match.groups()
            audio_info['duration'] = float(hours) * 3600 + float(minutes) * 60 + float(seconds)
        
        logging.info(f"Audio info for {file_path}: {audio_info}")
        return audio_info
    except Exception as e:
        logging.error(f"Error getting audio info: {e}")
        return audio_info

def is_valid_audio(file_path, min_duration_sec=5):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        logging.error(f"File does not exist or is empty: {file_path}")
        return False
    try:
        info = mediainfo(file_path)
        duration = float(info.get('duration', 0))
        if duration < min_duration_sec:
            logging.error(f"File too short (<{min_duration_sec}s): {file_path} (duration: {duration:.2f}s)")
            return False
    except Exception as e:
        logging.error(f"Could not get audio info for {file_path}: {e}")
        return False
    return True

def normalize_audio_ffmpeg(input_file, output_file, target_lufs=-14.0, apply_limiter=True):
    if not is_valid_audio(input_file):
        logging.error(f"Skipping normalization for invalid input: {input_file}")
        return False
    limiter_filter = ":peak=0.97" if apply_limiter else ""
    
    command = [
        "ffmpeg", "-y", "-i", input_file,
        "-af", f"loudnorm=I={target_lufs}:TP=-1:LRA=11{limiter_filter}",
        "-ar", "44100", output_file
    ]
    
    return run_ffmpeg_command(command, f"Normalizing audio to {target_lufs} LUFS")

def resample_audio_ffmpeg(input_file, output_file, sample_rate=44100):
    if not is_valid_audio(input_file):
        logging.error(f"Skipping resampling for invalid input: {input_file}")
        return False
    command = [
        "ffmpeg", "-y", "-i", input_file,
        "-ar", str(sample_rate), output_file
    ]
    
    return run_ffmpeg_command(command, f"Resampling audio to {sample_rate} Hz")

def mix_to_stereo_ffmpeg(left_file, right_file, output_file, left_pan=-0.2, right_pan=0.2, append_tones=False):
    if not is_valid_audio(left_file) or not is_valid_audio(right_file):
        logging.error(f"Skipping stereo mix due to invalid input(s): {left_file}, {right_file}")
        return False
    # First, ensure both files are at the same sample rate (44.1kHz)
    left_info = get_audio_info(left_file)
    right_info = get_audio_info(right_file)
    
    temp_files = []
    
    # If sample rate is not 44100, resample
    left_resampled = left_file
    if left_info['sample_rate'] != 44100:
        left_resampled = tempfile.mktemp(suffix='.wav')
        temp_files.append(left_resampled)
        resample_audio_ffmpeg(left_file, left_resampled)
    
    right_resampled = right_file
    if right_info['sample_rate'] != 44100:
        right_resampled = tempfile.mktemp(suffix='.wav')
        temp_files.append(right_resampled)
        resample_audio_ffmpeg(right_file, right_resampled)
    
    # Mix audio using FFmpeg's amerge and pan filters
    filter_complex = f"[0:a]pan=stereo|c0={left_pan}*c0+{right_pan}*c1|c1={left_pan}*c0+{right_pan}*c1[left];[1:a]pan=stereo|c0={right_pan}*c0+{left_pan}*c1|c1={right_pan}*c0+{left_pan}*c1[right];[left][right]amerge=inputs=2,pan=stereo|c0=c0+c2|c1=c1+c3[out]"
    
    command = [
        "ffmpeg", "-y", 
        "-i", left_resampled,
        "-i", right_resampled,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        output_file
    ]
    
    success = run_ffmpeg_command(command, "Mixing stereo channels")
    
    # Append tones if requested
    if success and append_tones and os.path.exists("tones.wav"):
        temp_output = tempfile.mktemp(suffix='.wav')
        temp_files.append(temp_output)
        
        # Move the mixed file to temp
        shutil.copy2(output_file, temp_output)
        
        # Concatenate with tones
        command = [
            "ffmpeg", "-y",
            "-i", temp_output,
            "-i", "tones.wav",
            "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[out]",
            "-map", "[out]",
            output_file
        ]
        
        success = run_ffmpeg_command(command, "Appending tones")
    
    # Clean up temp files
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    return success

def sanitize_text(text):
    sanitized = re.sub(r'[^\w\s]', '', text)
    return sanitized.replace(" ", "_")

def extract_first_words(lab_path, word_limit=12):
    with open(lab_path, 'r', encoding="utf-8") as lab_file:
        words = lab_file.read().split()[:word_limit]
    return sanitize_text(" ".join(words))

def run_command(command, description):
    try:
        subprocess.run(command, check=True)
        logging.info(f"{description} completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"{description} failed with error: {e}")
    except Exception as ex:
        logging.error(f"Unexpected error during {description}: {ex}")

def slicing(input_dir, output_dir):
    logging.info(f"Starting slicing for directory: {input_dir}")
    run_command(['fap', 'slice-audio-v2', '--no-merge-short', '--min-duration', '3', input_dir, output_dir], "Slicing")
    if not os.listdir(output_dir):
        logging.error("Slicing step completed, but no output files were created.")
        exit(1)
    logging.info(f"Slicing completed, files are available in {output_dir}")

def transcribe_directory(input_dir, num_workers=2):
    logging.info(f"Transcribing all files in directory: {input_dir} with {num_workers} workers")
    run_command(['fap', 'transcribe', '--lang', 'en', '--recursive', input_dir, '--num-workers', str(num_workers)], "Transcription")

def rename_and_copy_sliced_files(input_dir, target_dir, side):
    MAX_FILENAME_LENGTH = 128  # Maximum allowed filename length
    for root, dirs, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        audio_target_dir = os.path.join(target_dir, side, rel_path)
        os.makedirs(audio_target_dir, exist_ok=True)

        for filename in files:
            source_path = os.path.join(root, filename)

            if filename.endswith(".wav"):
                base_name = filename.split('.')[0]
                lab_filename = f"{base_name}.lab"
                lab_path = os.path.join(root, lab_filename)

                if os.path.exists(lab_path):
                    text_excerpt = extract_first_words(lab_path)
                    new_filename = f"{base_name}_{text_excerpt}.wav"
                else:
                    new_filename = filename

                # Ensure the new filename does not exceed the maximum length
                if len(new_filename) > MAX_FILENAME_LENGTH:
                    # Calculate how many characters to truncate
                    excess_length = len(new_filename) - MAX_FILENAME_LENGTH
                    # Truncate the text excerpt first
                    if '_' in new_filename:
                        base_name, text_excerpt_with_ext = new_filename.rsplit('_', 1)
                        text_excerpt, ext = text_excerpt_with_ext.rsplit('.', 1)
                        if len(text_excerpt) > excess_length:
                            text_excerpt = text_excerpt[:len(text_excerpt) - excess_length]
                        else:
                            text_excerpt = text_excerpt[:1]  # Ensure at least one character remains
                        new_filename = f"{base_name}_{text_excerpt}.{ext}"
                    else:
                        # If no underscore, truncate the base name
                        base_name = base_name[:len(base_name) - excess_length]
                        new_filename = f"{base_name}.wav"

                target_path = os.path.join(audio_target_dir, new_filename)
                shutil.copy2(source_path, target_path)
                logging.info(f"Sliced audio file copied to {target_path}")

            elif filename.endswith(".lab"):
                target_path = os.path.join(audio_target_dir, filename)
                shutil.copy2(source_path, target_path)
                logging.info(f".lab file copied to {target_path}")

def process_audio_files(input_dir, output_dir, transcribe_left=False, transcribe_right=False, 
                       append_tones=False, normalize_audio=False, target_lufs=-14.0, 
                       target_sample_rate=44100, left_pan=-0.2, right_pan=0.2, num_workers=2):
    logging.info("Starting audio processing")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    normalized_dir = os.path.join(run_output_dir, 'normalized')
    left_dir = os.path.join(run_output_dir, 'left')
    right_dir = os.path.join(run_output_dir, 'right')
    stereo_dir = os.path.join(run_output_dir, 'stereo')
    transcribed_and_sliced_dir = os.path.join(run_output_dir, 'transcribed-and-sliced')
    os.makedirs(normalized_dir, exist_ok=True)
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)
    os.makedirs(stereo_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if not os.path.isfile(file_path):
            continue  # Skip directories
        if not is_valid_audio(file_path):
            logging.error(f"Skipping invalid or too short audio file: {file_path}")
            continue
        file_timestamp = os.path.getmtime(file_path)
        logging.info(f"Processing file: {filename}")
        
        try:
            normalized_path = os.path.join(normalized_dir, filename)
            
            if normalize_audio:
                normalize_audio_ffmpeg(file_path, normalized_path, target_lufs)
            else:
                # Just copy the file or resample it if needed
                audio_info = get_audio_info(file_path)
                if audio_info['sample_rate'] != target_sample_rate:
                    resample_audio_ffmpeg(file_path, normalized_path, target_sample_rate)
                else:
                    shutil.copy2(file_path, normalized_path)
            
            os.utime(normalized_path, (file_timestamp, file_timestamp))
            logging.info(f"Processed audio saved to {normalized_path}")

        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")
            continue

    channel_pairs = identify_channel_pairs(normalized_dir)
    process_channel_pairs(channel_pairs, left_dir, right_dir, stereo_dir, 
                          append_tones, target_sample_rate, left_pan, right_pan)

    if transcribe_left:
        slicing(left_dir, left_dir)
        transcribe_directory(left_dir, num_workers=num_workers)
        rename_and_copy_sliced_files(left_dir, transcribed_and_sliced_dir, 'left')

    if transcribe_right:
        slicing(right_dir, right_dir)
        transcribe_directory(right_dir, num_workers=num_workers)
        rename_and_copy_sliced_files(right_dir, transcribed_and_sliced_dir, 'right')

def identify_channel_pairs(normalized_dir):
    pairs = {}
    for filename in os.listdir(normalized_dir):
        base_key = filename.replace('trans_', '').replace('recv_', '')

        if filename.startswith('recv_'):
            pairs.setdefault(base_key, {})['left'] = os.path.join(normalized_dir, filename)
        elif filename.startswith('trans_'):
            pairs.setdefault(base_key, {})['right'] = os.path.join(normalized_dir, filename)

    return pairs

def process_channel_pairs(pairs, left_dir, right_dir, stereo_dir, append_tones, 
                          target_sample_rate=44100, left_pan=-0.2, right_pan=0.2):
    for base_name, channels in pairs.items():
        left_path = channels.get('left')
        right_path = channels.get('right')

        if left_path:
            save_channel_audio(left_path, left_dir, target_sample_rate)
        if right_path:
            save_channel_audio(right_path, right_dir, target_sample_rate)
        
        if left_path and right_path:
            stereo_output_path = os.path.join(stereo_dir, f"{base_name}_stereo.wav")
            mix_to_stereo_ffmpeg(left_path, right_path, stereo_output_path, 
                                left_pan, right_pan, append_tones)
            
            # Preserve timestamp
            if os.path.exists(left_path):
                os.utime(stereo_output_path, (os.path.getmtime(left_path), os.path.getmtime(left_path)))

def save_channel_audio(audio_path, target_dir, target_sample_rate=44100):
    filename = os.path.basename(audio_path)
    output_path = os.path.join(target_dir, filename)
    
    # Get audio info
    audio_info = get_audio_info(audio_path)
    
    # If sample rate needs changing, resample
    if audio_info['sample_rate'] != target_sample_rate:
        resample_audio_ffmpeg(audio_path, output_path, target_sample_rate)
    else:
        shutil.copy2(audio_path, output_path)
    
    # Preserve timestamp
    os.utime(output_path, (os.path.getmtime(audio_path), os.path.getmtime(audio_path)))
    logging.info(f"Channel audio saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files with FFmpeg for proper resampling, normalization, and stereo mixing.")
    parser.add_argument("--input-dir", required=True, help="Directory containing input audio files")
    parser.add_argument("--output-dir", required=True, help="Directory where output will be saved")
    parser.add_argument("--transcribe", action="store_true", help="Enable transcription for both left and right channels")
    parser.add_argument("--transcribe_left", action="store_true", help="Enable transcription for left channel only")
    parser.add_argument("--transcribe_right", action="store_true", help="Enable transcription for right channel only")
    parser.add_argument("--tones", action="store_true", help="Append 'tones.wav' to the end of stereo output files")
    parser.add_argument("--normalize", action="store_true", help="Enable loudness normalization")
    parser.add_argument("--target-lufs", type=float, default=-14.0, help="Target loudness in LUFS (default: -14)")
    parser.add_argument("--target-sample-rate", type=int, default=44100, help="Target sample rate in Hz (default: 44100)")
    parser.add_argument("--left-pan", type=float, default=-0.2, help="Pan value for left channel (-1.0 to 1.0, default: -0.2)")
    parser.add_argument("--right-pan", type=float, default=0.2, help="Pan value for right channel (-1.0 to 1.0, default: 0.2)")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of workers for transcription (default: 2)")

    args = parser.parse_args()
    
    transcribe_left = args.transcribe or args.transcribe_left
    transcribe_right = args.transcribe or args.transcribe_right

    process_audio_files(
        args.input_dir,
        args.output_dir,
        transcribe_left=transcribe_left,
        transcribe_right=transcribe_right,
        append_tones=args.tones,
        normalize_audio=args.normalize,
        target_lufs=args.target_lufs,
        target_sample_rate=args.target_sample_rate,
        left_pan=args.left_pan,
        right_pan=args.right_pan,
        num_workers=args.num_workers
    )
