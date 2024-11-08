
import argparse
import os
import sys
import subprocess
import yt_dlp
import shutil
import logging
from datetime import datetime

logging.basicConfig(level=logging.DEBUG)

def download_audio(url, download_dir):
    """Download audio from URL and save it in the specified download directory"""
    temp_audio_file_path = os.path.join(download_dir, 'downloaded_audio.m4a')
    
    ydl_opts = {
        'format': '140',  # Force M4A format
        'outtmpl': temp_audio_file_path,  # Output directly to download directory
        'quiet': True
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        logging.error(f"Error downloading audio: {e}")
        sys.exit(1)
    
    if os.path.getsize(temp_audio_file_path) == 0:
        logging.error("Error: Downloaded file is empty.")
        sys.exit(1)

    return download_dir  # Return directory path containing the downloaded audio file

def run_command(command, stage_name, stage_dir):
    """Run a subprocess command and log progress by stage"""
    logging.info(f"Running stage: {stage_name}")
    os.makedirs(stage_dir, exist_ok=True)

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during {stage_name} execution: {e}")
        sys.exit(1)

    return stage_dir  # Return the directory for the processing stage

def copy_and_rename_transcription_files(slicing_subfolder, transcription_dir):
    """Copy .wav and .lab files from slicing subfolder to transcription directory with renaming based on .lab content"""
    os.makedirs(transcription_dir, exist_ok=True)
    for i, file in enumerate(sorted(os.listdir(slicing_subfolder))):
        file_path = os.path.join(slicing_subfolder, file)
        if file.endswith(".wav"):
            # Find matching .lab file for renaming
            lab_file_name = file.replace(".wav", ".lab")
            lab_file_path = os.path.join(slicing_subfolder, lab_file_name)

            if os.path.exists(lab_file_path):
                # Read first 2-8 words from the .lab file
                with open(lab_file_path, 'r') as lab_file:
                    transcription_text = lab_file.readline().strip()
                words = "_".join(transcription_text.split()[:8])
                if not words:  # If there are no words, fallback to the numbered prefix only
                    dest_file_name = f"{i:04d}.wav"
                    dest_lab_name = f"{i:04d}.lab"
                else:
                    dest_file_name = f"{i:04d}_{words}.wav"
                    dest_lab_name = f"{i:04d}_{words}.lab"

                # Copy and rename .wav and .lab files to transcription directory
                shutil.copy(file_path, os.path.join(transcription_dir, dest_file_name))
                shutil.copy(lab_file_path, os.path.join(transcription_dir, dest_lab_name))
                logging.info(f"Copied and renamed {file} to {dest_file_name} and {lab_file_name} to {dest_lab_name}")

def get_oldest_file_timestamp(input_dir):
    """Get the timestamp of the oldest file in the directory"""
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    oldest_file = min(files, key=os.path.getmtime)
    return datetime.fromtimestamp(os.path.getmtime(oldest_file)).strftime('%Y%m%d_%H%M%S')

def main():
    parser = argparse.ArgumentParser(description="Efficient audio processing script with structured outputs.")
    parser.add_argument('--url', type=str, help='URL to download audio from')
    parser.add_argument('--output_dir', type=str, default="output", help='Directory for storing output files')
    parser.add_argument('input_dir', nargs='?', help='Directory of input files if no URL is provided')
    args = parser.parse_args()

    # Determine the main output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    main_output_dir = os.path.join(args.output_dir, f"output_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)

    # Step 1: Determine input directory or download if URL provided
    input_dir = args.input_dir
    if args.url:
        input_dir = download_audio(args.url, main_output_dir)

    if not input_dir or not os.path.isdir(input_dir):
        logging.error("Error: No valid input directory provided.")
        sys.exit(1)

    # Step 2: WAV Conversion Stage (Check cache)
    wav_output_dir = os.path.join(main_output_dir, "wav_conversion")
    os.makedirs(wav_output_dir, exist_ok=True)
    if not os.listdir(wav_output_dir):  # Proceed if folder is empty
        logging.info("Starting WAV conversion stage.")
        run_command(['fap', 'to-wav', '--recursive', input_dir, wav_output_dir], "WAV Conversion", wav_output_dir)
    else:
        logging.info("Using cached WAV files from wav_conversion folder.")

    # Step 3: Loudness Normalization Stage
    norm_output_dir = os.path.join(main_output_dir, "loudness_normalization")
    os.makedirs(norm_output_dir, exist_ok=True)
    if not os.listdir(norm_output_dir):
        logging.info("Starting loudness normalization stage.")
        run_command(['fap', 'loudness-norm', '--recursive', wav_output_dir, norm_output_dir], "Loudness Normalization", norm_output_dir)
    else:
        logging.info("Using cached normalized files from loudness_normalization folder.")

    # Step 4: Audio Slicing Stage
    slice_output_dir = os.path.join(main_output_dir, "slicing")
    os.makedirs(slice_output_dir, exist_ok=True)
    if not os.listdir(slice_output_dir):
        logging.info("Starting audio slicing stage.")
        run_command(['fap', 'slice-audio-v2', '--no-merge-short', '--min-duration', '3', norm_output_dir, slice_output_dir], "Slicing", slice_output_dir)
    else:
        logging.info("Using cached sliced files from slicing folder.")

    # Step 5: Transcription Stage
    slicing_subfolder = os.path.join(slice_output_dir, "downloaded_audio_0000")
    transcribe_output_dir = os.path.join(main_output_dir, "transcription")
    os.makedirs(transcribe_output_dir, exist_ok=True)
    if not os.listdir(transcribe_output_dir):
        logging.info("Starting transcription stage.")
        run_command(['fap', 'transcribe', '--lang', 'en', '--recursive', slicing_subfolder], "Transcription", transcribe_output_dir)
        
        # Copy and rename transcription files from slicing subfolder to transcription output folder
        copy_and_rename_transcription_files(slicing_subfolder, transcribe_output_dir)
    else:
        logging.info("Using cached transcription files from transcription folder.")

    logging.info(f"Processing complete. Outputs are in {main_output_dir}")

if __name__ == "__main__":
    main()
