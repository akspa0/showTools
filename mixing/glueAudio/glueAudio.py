import os
import re
import argparse
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

# Function to ensure the full directory exists before writing files
def ensure_directory_exists(file_path):
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

# Function to check if the filename has a valid prefix (1-9999 followed by a dash)
def is_valid_audio_prefix(filename):
    # Regex to check if filename starts with a number between 1 and 9999 followed by '-'
    match = re.match(r'^\d{1,4}-', filename)
    return bool(match)

# Function to merge audio files in each subfolder
def merge_audio_in_subfolders(input_folder, output_folder, use_prefix):
    # Walk through each subfolder in the input directory
    for root, _, files in os.walk(input_folder):
        valid_files = []
        
        # Identify all valid audio files based on the prefix rule or process all
        for filename in sorted(files):
            if filename.endswith(('.wav', '.mp3', '.ogg', '.flac')):
                if use_prefix and is_valid_audio_prefix(filename):
                    valid_files.append(os.path.join(root, filename))
                elif not use_prefix:
                    valid_files.append(os.path.join(root, filename))
                print(f"Valid file: {os.path.join(root, filename)}")

        # Skip if no valid files in the folder
        if not valid_files:
            print(f"No valid files found in {root}, skipping...")
            continue

        # Merge the valid audio files
        merged = AudioSegment.empty()
        for file_path in valid_files:
            try:
                audio = AudioSegment.from_file(file_path)
                merged += audio
            except CouldntDecodeError:
                print(f"Warning: Could not decode {file_path}, skipping this file.")

        # Create the relative path for the output folder
        relative_path = os.path.relpath(root, input_folder)
        output_dir = os.path.join(output_folder, relative_path)

        # Ensure the output directory exists
        ensure_directory_exists(output_dir)

        # Name the output file based on the current subfolder
        output_file = os.path.join(output_dir, "merged_audio.wav")
        print(f"Exporting merged audio to: {output_file}")

        # Ensure that the output directory is created before exporting
        ensure_directory_exists(output_file)

        # Export the merged audio file
        merged.export(output_file, format="wav")
        print(f"Merged audio exported to {output_file}")

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Merge audio files from subfolders into merged files.")
    parser.add_argument('input_folder', type=str, help="The root folder containing subfolders with audio files.")
    parser.add_argument('output_folder', type=str, help="The root folder where merged audio files will be saved.")
    parser.add_argument('--use-prefix', action='store_true', help="Only merge audio files with a 1-9999 prefix (default: process all audio files).")

    # Parse the arguments
    args = parser.parse_args()

    # Check if the input folder exists
    if not os.path.isdir(args.input_folder):
        print(f"Error: The folder '{args.input_folder}' does not exist.")
        exit(1)

    # Merge the audio files from subfolders
    merge_audio_in_subfolders(args.input_folder, args.output_folder, args.use_prefix)
