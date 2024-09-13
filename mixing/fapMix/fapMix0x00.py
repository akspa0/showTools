import os
import argparse
import subprocess
import shutil
import tempfile
import logging

# Setup logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def run_command(command):
    """Helper function to run a subprocess command and check for errors."""
    logging.debug(f"Running command: {' '.join(command)}")
    subprocess.run(command, check=True)

def copy_wav_files(input_dir, wav_output_dir):
    """Copy existing WAV files from input directory to wav_output_dir."""
    if not os.path.exists(wav_output_dir):
        os.makedirs(wav_output_dir)
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                src_file = os.path.join(root, file)
                dst_file = os.path.join(wav_output_dir, file)
                logging.debug(f"Copying WAV file: {src_file} to {dst_file}")
                shutil.copy2(src_file, dst_file)

def convert_to_wav_if_needed(input_dir, wav_output_dir):
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
        copy_wav_files(input_dir, wav_output_dir)

def extract_first_chars_from_lab(lab_file_path, char_limit=40):
    """Extract the first 'char_limit' characters from the .lab file."""
    with open(lab_file_path, 'r') as file:
        content = file.read().strip()
    
    return content[:char_limit]

def rename_and_copy_files(output_dir, temp_dir):
    """Rename and copy WAV files with transcription text and include .lab files."""
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
                    
                    # Copy .lab file to the same sub-folder
                    new_lab_file_path = os.path.join(output_subfolder, file)
                    logging.debug(f"Copying .lab file: {lab_file_path} to {new_lab_file_path}")
                    shutil.copy2(lab_file_path, new_lab_file_path)

def check_for_lab_files(temp_dir):
    """Check if any .lab files are present in the temp_dir after transcription."""
    lab_files_found = False
    for root, dirs, files in os.walk(temp_dir):
        if any(file.endswith(".lab") for file in files):
            logging.debug(f".lab files found in {root}")
            lab_files_found = True
    
    if not lab_files_found:
        logging.error(f"No .lab files found in {temp_dir}. Transcription may have failed.")

def main(input_dir, output_dir):
    # Create the output directory if it doesn't exist
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
        convert_to_wav_if_needed(input_dir, wav_output_dir)

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
        rename_and_copy_files(output_dir, slice_output_dir)

    finally:
        # Keeping temporary files for inspection instead of deleting
        logging.debug(f"Temporary files are stored in {temp_dir}. Please inspect manually.")
        print(f"Temporary files are saved in: {temp_dir}")

if __name__ == "__main__":
    # Argument parser for input and output directories
    parser = argparse.ArgumentParser(description="Process audio files with fap tools.")
    parser.add_argument("input_dir", help="The input directory containing audio files.")
    parser.add_argument("output_dir", help="The output directory to save the processed files.")

    args = parser.parse_args()

    # Run the main function
    main(args.input_dir, args.output_dir)
