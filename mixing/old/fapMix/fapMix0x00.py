import os
import argparse
import subprocess
import shutil
import tempfile
import logging
from datetime import datetime
import zipfile

# Setup logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def run_command(command):
    """Helper function to run a subprocess command and check for errors."""
    logging.debug(f"Running command: {' '.join(command)}")
    subprocess.run(command, check=True)

def filter_files(input_dir, vocals_only, pbx_only):
    """Filter files based on the given options."""
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                if vocals_only:
                    if "(Vocals)" in file:
                        yield os.path.join(root, file)
                elif pbx_only:
                    if "recv_out" in file or "trans_out" in file:
                        yield os.path.join(root, file)
                else:
                    yield os.path.join(root, file)

def copy_wav_files(input_dir, wav_output_dir, vocals_only, pbx_only):
    """Copy WAV files from input directory to wav_output_dir based on the filter."""
    if not os.path.exists(wav_output_dir):
        os.makedirs(wav_output_dir)

    for file_path in filter_files(input_dir, vocals_only, pbx_only):
        file_name = os.path.basename(file_path)
        dst_file = os.path.join(wav_output_dir, file_name)
        logging.debug(f"Copying WAV file: {file_path} to {dst_file}")
        shutil.copy2(file_path, dst_file)

def convert_to_wav_if_needed(input_dir, wav_output_dir, vocals_only, pbx_only):
    """Check if the input directory contains any WAV files, if not, convert to WAV."""
    wav_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
    
    if not wav_files:  # No WAV files found, convert to WAV
        logging.debug(f"No WAV files found in {input_dir}. Converting to WAV.")
        to_wav_command = [
            "fap", "to-wav",
            "--recursive",
            input_dir,
            wav_output_dir
        ]
        run_command(to_wav_command)
    else:  # Copy existing WAV files to wav_output_dir
        logging.debug(f"Found existing WAV files in {input_dir}. Copying them to {wav_output_dir}.")
        copy_wav_files(input_dir, wav_output_dir, vocals_only, pbx_only)

def extract_first_chars_from_lab(lab_file_path, char_limit=40):
    """Extract the first 'char_limit' characters from the .lab file."""
    with open(lab_file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read().strip()
    
    return content[:char_limit]

def rename_and_copy_files(output_dir, temp_dir, pbx_only):
    """Rename and copy WAV files with transcription text and include .lab files."""
    lab_dataset_dir = os.path.join(output_dir, "lab_dataset")
    if not os.path.exists(lab_dataset_dir):
        os.makedirs(lab_dataset_dir)
    
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.endswith(".lab"):
                lab_file_path = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]
                wav_file_path = os.path.join(root, f"{base_name}.wav")
                
                if os.path.exists(wav_file_path):
                    transcription_text = extract_first_chars_from_lab(lab_file_path)
                    safe_text = ''.join(c if c.isalnum() or c.isspace() else '_' for c in transcription_text)  # Sanitize text
                    new_wav_file_name = f"{base_name}-{safe_text}.wav"
                    
                    # Maintain sub-folder structure
                    relative_path = os.path.relpath(root, temp_dir)
                    output_subfolder = os.path.join(output_dir, relative_path)
                    if not os.path.exists(output_subfolder):
                        os.makedirs(output_subfolder)
                    
                    # Copy renamed .wav file
                    new_wav_file_path = os.path.join(output_subfolder, new_wav_file_name)
                    logging.debug(f"Copying and renaming file: {wav_file_path} to {new_wav_file_path}")
                    shutil.copy2(wav_file_path, new_wav_file_path)
                    
                    # Copy .lab file to the lab dataset sub-folder
                    new_lab_file_path = os.path.join(lab_dataset_dir, file)
                    logging.debug(f"Copying .lab file: {lab_file_path} to {new_lab_file_path}")
                    shutil.copy2(lab_file_path, new_lab_file_path)

                # Copy original WAV file to output directory if not processed
                if not pbx_only:
                    original_wav_file_path = os.path.join(output_dir, os.path.basename(wav_file_path))
                    logging.debug(f"Copying original WAV file: {wav_file_path} to {original_wav_file_path}")
                    shutil.copy2(wav_file_path, original_wav_file_path)

def check_for_lab_files(temp_dir):
    """Check if any .lab files are present in the temp_dir after transcription."""
    lab_files_found = False
    for root, dirs, files in os.walk(temp_dir):
        if any(file.endswith(".lab") for file in files):
            logging.debug(f".lab files found in {root}")
            lab_files_found = True
    
    if not lab_files_found:
        logging.error(f"No .lab files found in {temp_dir}. Transcription may have failed.")

def find_oldest_timestamp(input_dir):
    """Find the oldest timestamp from files in the input directory."""
    oldest_timestamp = None
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                timestamp = os.path.getmtime(file_path)  # Get modification time
                if oldest_timestamp is None or timestamp < oldest_timestamp:
                    oldest_timestamp = timestamp
            except Exception as e:
                logging.error(f"Error getting timestamp for {file_path}: {e}")
    
    return oldest_timestamp

def generate_output_folder_name(input_dir):
    """Generate a unique output folder name based on the oldest file timestamp."""
    oldest_timestamp = find_oldest_timestamp(input_dir)
    
    if oldest_timestamp:
        date_time = datetime.fromtimestamp(oldest_timestamp)
        month_name = date_time.strftime("%B")  # Full month name
        folder_name = f"{date_time.day:02d}{month_name}{date_time.year}-snippets"
        return folder_name
    else:
        # Fallback if no timestamp found
        folder_name = f"UnknownDate-snippets"
        logging.warning("No valid timestamps found in input files. Using default folder name.")
        return folder_name

def zip_output_folder(output_dir):
    """Zip the output folder."""
    zip_file_name = f"{output_dir}.zip"
    logging.debug(f"Creating zip file: {zip_file_name}")
    
    with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=output_dir)
                logging.debug(f"Adding file to zip: {file_path}")
                zipf.write(file_path, arcname)

def zip_lab_dataset(lab_dataset_dir):
    """Zip the lab dataset folder."""
    zip_file_name = f"{lab_dataset_dir}.zip"
    logging.debug(f"Creating lab dataset zip file: {zip_file_name}")
    
    with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(lab_dataset_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=lab_dataset_dir)
                logging.debug(f"Adding file to lab dataset zip: {file_path}")
                zipf.write(file_path, arcname)

def main(input_dir, zip_output, vocals_only, pbx_only):
    # Generate output directory name and create it
    output_dir = generate_output_folder_name(input_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create temporary directories for intermediate files
    temp_dir = tempfile.mkdtemp()
    wav_output_dir = os.path.join(temp_dir, "wav_files")
    norm_output_dir = os.path.join(temp_dir, "normalized_files")
    slice_output_dir = os.path.join(temp_dir, "sliced_files")

    try:
        # Convert non-WAV files to WAV or copy existing WAV files
        logging.debug(f"Starting WAV conversion/check for {input_dir}.")
        convert_to_wav_if_needed(input_dir, wav_output_dir, vocals_only, pbx_only)

        # Step 1: Normalize loudness (output to norm_output_dir)
        logging.debug("Running loudness normalization.")
        loudness_norm_command = [
            "fap", "loudness-norm",
            "--recursive",
            wav_output_dir,
            norm_output_dir
        ]
        run_command(loudness_norm_command)

        # Step 2: Slice audio files (output to slice_output_dir)
        logging.debug("Running audio slicing.")
        slice_command = [
            "fap", "slice-audio-v2",
            "--no-merge-short",
            "--min-duration", "3",
            norm_output_dir,
            slice_output_dir
        ]
        run_command(slice_command)

        # Step 3: Transcribe audio files
        logging.debug("Running transcription.")
        transcribe_command = [
            "fap", "transcribe",
            "--lang", "en",
            "--recursive",
            slice_output_dir
        ]
        run_command(transcribe_command)

        # Check for .lab files in the slice_output_dir
        check_for_lab_files(slice_output_dir)

        # Rename and copy WAV files with transcription text, and include .lab files
        logging.debug("Renaming and copying WAV files with transcription text.")
        rename_and_copy_files(output_dir, slice_output_dir, pbx_only)

        # Zip the output folder if specified
        if zip_output:
            zip_output_folder(output_dir)
        
        # Zip the lab dataset folder
        lab_dataset_dir = os.path.join(output_dir, "lab_dataset")
        if os.path.exists(lab_dataset_dir):
            zip_lab_dataset(lab_dataset_dir)

    finally:
        # Keeping temporary files for inspection instead of deleting
        logging.debug(f"Temporary files are stored in {temp_dir}. Please inspect manually.")
        print(f"Temporary files are saved in: {temp_dir}")

if __name__ == "__main__":
    # Argument parser for input directory and options
    parser = argparse.ArgumentParser(description="Process audio files and transcripts.")
    parser.add_argument("input_dir", help="Directory containing input audio files.")
    parser.add_argument("--zip", action="store_true", help="Zip the output folder.")
    parser.add_argument("--vocals-only", action="store_true", help="Process only files with '(Vocals)' in the filename.")
    parser.add_argument("--pbx", action="store_true", help="Process only files named 'recv_out' and 'trans_out'.")
    args = parser.parse_args()
    
    main(args.input_dir, args.zip, args.vocals_only, args.pbx)
