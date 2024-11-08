
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
    """Download audio from URL and save in download directory"""
    temp_audio_file_path = os.path.join(download_dir, 'downloaded_audio.m4a')
    
    ydl_opts = {
        'format': '140',
        'outtmpl': temp_audio_file_path,
        'quiet': True
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        logging.error(f"Error downloading audio: {e}")
        sys.exit(1)
    
    if os.path.getsize(temp_audio_file_path) == 0:
        logging.error("Downloaded file is empty.")
        sys.exit(1)

    return download_dir

def run_command(command, stage_name):
    """Run a subprocess command and log progress"""
    logging.info(f"Running stage: {stage_name}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during {stage_name} execution: {e}")
        sys.exit(1)

def wav_conversion(input_dir, output_dir):
    """Convert audio files to WAV format"""
    run_command(['fap', 'to-wav', '--recursive', input_dir, output_dir], "WAV Conversion")

def separation(input_dir, output_dir):
    """Separate WAV files into vocal components"""
    run_command(['fap', 'separate', input_dir, output_dir], "Separation")
    if not os.listdir(output_dir):
        logging.error("Separation step completed, but no output files were created.")
        sys.exit(1)
    logging.info(f"Separation completed successfully, files are available in {output_dir}")

def slicing(input_dir, output_dir):
    """Slice audio files into smaller segments for transcription"""
    run_command(['fap', 'slice-audio-v2', '--no-merge-short', '--min-duration', '3', input_dir, output_dir], "Slicing")
    if not os.listdir(output_dir):
        logging.error("Slicing step completed, but no output files were created.")
        sys.exit(1)
    logging.info(f"Slicing completed, files are available in {output_dir}")

def transcribe(input_dir):
    """Transcribe each sliced audio segment into .lab files"""
    run_command(['fap', 'transcribe', '--lang', 'en', '--recursive', input_dir], "Transcription")
    logging.info(f"Transcription completed, files are available in {input_dir}")

def sanitize_filename(words, prefix, max_length=128):
    """Sanitize filename by removing unwanted characters, replacing spaces with underscores, and limiting length"""
    sanitized_words = "_".join(words.split())[:max_length]  # Replace spaces with underscores and limit length
    sanitized = "".join(char if char.isalnum() or char == "_" else "" for char in sanitized_words)
    return f"{prefix}_{sanitized[:max_length]}"

def rename_and_copy_transcription_files(transcription_dir, final_output_dir):
    """Rename transcription files based on the content in their corresponding .lab files and copy to final output"""
    os.makedirs(final_output_dir, exist_ok=True)
    for i, file in enumerate(sorted(os.listdir(transcription_dir))):
        if file.endswith(".wav"):
            lab_file_path = os.path.join(transcription_dir, file.replace(".wav", ".lab"))
            prefix = f"{i:04d}"

            if os.path.exists(lab_file_path):
                with open(lab_file_path, 'r') as lab_file:
                    transcription_text = lab_file.readline().strip()
                words = "_".join(transcription_text.split()[:10]) if transcription_text else ""  # First 10 words

                # Sanitize and limit filename
                dest_file_name = sanitize_filename(words, prefix) if words else prefix
                dest_wav_path = os.path.join(final_output_dir, dest_file_name + ".wav")
                dest_lab_path = os.path.join(final_output_dir, dest_file_name + ".lab")

                # Copy and rename both .wav and .lab files
                shutil.copy(os.path.join(transcription_dir, file), dest_wav_path)
                shutil.copy(lab_file_path, dest_lab_path)
                logging.info(f"Copied and renamed {file} to {dest_file_name}.wav and {lab_file_path} to {dest_file_name}.lab")
            else:
                logging.warning(f"Matching .lab file for {file} not found.")

def main():
    parser = argparse.ArgumentParser(description="Streamlined audio processing script.")
    parser.add_argument('--url', type=str, help='URL to download audio from')
    parser.add_argument('--output_dir', type=str, default="output", help='Directory for storing output files')
    parser.add_argument('input_dir', nargs='?', help='Directory of input files if no URL is provided')
    args = parser.parse_args()

    # Create main output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    main_output_dir = os.path.join(args.output_dir, f"output_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)

    # Step 1: Acquire Input Audio
    input_dir = args.input_dir
    if args.url:
        input_dir = download_audio(args.url, main_output_dir)

    if not input_dir or not os.path.isdir(input_dir):
        logging.error("No valid input directory provided.")
        sys.exit(1)

    # Step 2: WAV Conversion
    wav_output_dir = os.path.join(main_output_dir, "wav_conversion")
    os.makedirs(wav_output_dir, exist_ok=True)
    wav_conversion(input_dir, wav_output_dir)

    # Step 3: Separation
    separation_output_dir = os.path.join(main_output_dir, "separation_output")
    os.makedirs(separation_output_dir, exist_ok=True)
    separation(wav_output_dir, separation_output_dir)

    # Step 4: Slicing
    slicing_output_dir = os.path.join(main_output_dir, "slicing_output")
    os.makedirs(slicing_output_dir, exist_ok=True)
    slicing(separation_output_dir, slicing_output_dir)

    # Locate the actual subfolder for transcription
    slicing_subfolder = None
    for subfolder in os.listdir(slicing_output_dir):
        subfolder_path = os.path.join(slicing_output_dir, subfolder)
        if os.path.isdir(subfolder_path):
            slicing_subfolder = subfolder_path
            break

    if not slicing_subfolder:
        logging.error("No subfolder found in slicing output for transcription.")
        sys.exit(1)

    # Step 5: Transcription
    transcribe(slicing_subfolder)

    # Step 6: Rename and Copy Transcription Files
    final_output_dir = os.path.join(main_output_dir, "final_output")
    rename_and_copy_transcription_files(slicing_subfolder, final_output_dir)

if __name__ == "__main__":
    main()
