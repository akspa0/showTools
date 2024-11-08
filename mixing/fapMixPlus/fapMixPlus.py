
import os
import sys
import subprocess
import shutil
from datetime import datetime
import logging

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
        import yt_dlp
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=True)
            title = result.get('title', '')
    except Exception as e:
        logging.error(f"Error downloading audio: {e}")
        sys.exit(1)
    
    if os.path.getsize(temp_audio_file_path) == 0:
        logging.error("Downloaded file is empty.")
        sys.exit(1)

    return download_dir, title  # Return the directory and the title

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

def sanitize_filename(name, max_length=128):
    """Sanitize filename by removing unwanted characters and limiting length"""
    sanitized = "".join(char if char.isalnum() or char == "_" else "_" for char in name)[:max_length]
    return sanitized

def rename_and_copy_transcription_files(slicing_output_subdir, final_output_dir):
    """Rename transcription files based on .lab file content and copy to final output"""
    os.makedirs(final_output_dir, exist_ok=True)
    for i, file in enumerate(sorted(os.listdir(slicing_output_subdir))):
        if file.endswith(".wav"):
            lab_file_path = os.path.join(slicing_output_subdir, file.replace(".wav", ".lab"))
            prefix = f"{i:04d}"

            # Check if the corresponding .lab file exists
            if os.path.exists(lab_file_path):
                with open(lab_file_path, 'r') as lab_file:
                    transcription_text = lab_file.readline().strip()
                words = "_".join(transcription_text.split()[:10]) if transcription_text else ""

                # Sanitize and create final filenames
                dest_file_name = sanitize_filename(words, 128) if words else prefix
                dest_wav_path = os.path.join(final_output_dir, dest_file_name + ".wav")
                dest_lab_path = os.path.join(final_output_dir, dest_file_name + ".lab")

                # Copy and rename both .wav and .lab files to final_output
                shutil.copy(os.path.join(slicing_output_subdir, file), dest_wav_path)
                shutil.copy(lab_file_path, dest_lab_path)
                logging.info(f"Copied and renamed {file} to {dest_file_name}.wav and {lab_file_path} to {dest_file_name}.lab")
            else:
                logging.warning(f"Matching .lab file for {file} not found in {slicing_output_subdir}. Skipping.")

def get_oldest_file_date(input_dir):
    """Get date of the oldest file in the input directory in DDMonthYYYY format"""
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    if files:
        oldest_file = min(files, key=os.path.getmtime)
        file_time = datetime.fromtimestamp(os.path.getmtime(oldest_file))
    else:
        file_time = datetime.now()
    return file_time.strftime("%d%B%Y")

def zip_final_output(final_output_dir, output_dir, zip_name):
    """Zip the final_output directory with the specified zip name"""
    zip_path = os.path.join(output_dir, f"{zip_name}.zip")
    shutil.make_archive(zip_path.replace(".zip", ""), 'zip', final_output_dir)
    logging.info(f"Final output zipped as {zip_path}")
    return zip_path

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Streamlined audio processing script.")
    parser.add_argument('--url', type=str, help='URL to download audio from')
    parser.add_argument('--output_dir', type=str, default="output", help='Directory for storing output files')
    parser.add_argument('input_dir', nargs='?', help='Directory of input files if no URL is provided')
    args = parser.parse_args()

    # Set up output directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    main_output_dir = os.path.join(args.output_dir, f"output_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)

    # Step 1: Acquire Input Audio
    input_dir = args.input_dir
    identifier = ""  # This will hold our identifier suffix
    if args.url:
        input_dir, identifier = download_audio(args.url, main_output_dir)
    elif args.output_dir:
        identifier = args.output_dir  # Use output directory name if provided

    identifier = sanitize_filename(identifier, 50)  # Sanitize and limit identifier length

    if not input_dir or not os.path.isdir(input_dir):
        logging.error("No valid input directory provided.")
        sys.exit(1)

    # Step 2-5: Process Audio
    wav_output_dir = os.path.join(main_output_dir, "wav_conversion")
    separation_output_dir = os.path.join(main_output_dir, "separation_output")
    slicing_output_dir = os.path.join(main_output_dir, "slicing_output")
    final_output_dir = os.path.join(main_output_dir, "final_output")
    os.makedirs(final_output_dir, exist_ok=True)

    wav_conversion(input_dir, wav_output_dir)
    separation(wav_output_dir, separation_output_dir)
    slicing(separation_output_dir, slicing_output_dir)
    transcribe(slicing_output_dir)

    # Locate the actual subfolder for slicing output
    slicing_output_subdir = None
    for subfolder in os.listdir(slicing_output_dir):
        subfolder_path = os.path.join(slicing_output_dir, subfolder)
        if os.path.isdir(subfolder_path):
            slicing_output_subdir = subfolder_path
            break

    if not slicing_output_subdir:
        logging.error("No subfolder found in slicing output for renaming and copying to final output.")
        sys.exit(1)

    rename_and_copy_transcription_files(slicing_output_subdir, final_output_dir)

    # Step 6: Zip the final output with date + identifier format
    zip_date = get_oldest_file_date(input_dir)
    zip_name = f"{zip_date}-{identifier}" if identifier else zip_date
    zip_final_output(final_output_dir, main_output_dir, zip_name)

if __name__ == "__main__":
    main()
