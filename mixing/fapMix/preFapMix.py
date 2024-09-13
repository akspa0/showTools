import os
import re
import subprocess
import shutil
import tempfile
import logging

# Setup logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def run_command(command):
    """Helper function to run a subprocess command and check for errors."""
    logging.debug(f"Running command: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        if result.returncode != 0:
            logging.error(f"Command failed with error: {result.stderr}")
    except Exception as e:
        logging.error(f"Exception occurred: {e}")

def normalize_loudness(input_dir, norm_output_dir):
    """Normalize the loudness of audio files."""
    if not os.path.exists(norm_output_dir):
        os.makedirs(norm_output_dir)
    
    loudness_norm_command = [
        "fap", "loudness-norm",
        "--recursive",
        input_dir,
        norm_output_dir
    ]
    run_command(loudness_norm_command)

def mix_audio(input_dir, output_dir):
    """Mix audio files with 40% separation between channels."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_sets = {}
    
    # Regex to extract Unix timestamps from filenames
    timestamp_regex = re.compile(r'(\d{10,11})\.')

    for root, _, files in os.walk(input_dir):
        for file in files:
            match = timestamp_regex.search(file)
            if match:
                timestamp = match.group(1)
                if timestamp not in file_sets:
                    file_sets[timestamp] = {"left": None, "right": None}

                if "recv_out" in file or "_out" in file:
                    file_sets[timestamp]["left"] = os.path.join(root, file)
                elif "trans_out" in file or "_in" in file:
                    file_sets[timestamp]["right"] = os.path.join(root, file)
    
    for timestamp, files in file_sets.items():
        left_file = files.get("left")
        right_file = files.get("right")

        if left_file and right_file:
            output_file = os.path.join(output_dir, f"{timestamp}.mkv")
            mix_command = [
                "ffmpeg", "-i", left_file, "-i", right_file, "-filter_complex",
                "[0:a]pan=stereo|c0=c0|c1=c0[a1];[1:a]pan=stereo|c0=c0|c1=c1[a2];[a1][a2]amerge=inputs=2[a]",
                "-map", "[a]", "-c:a", "aac", "-b:a", "192k", output_file
            ]
            run_command(mix_command)
        else:
            logging.warning(f"Missing left or right channel for timestamp {timestamp}. Skipping.")

def main(input_dir):
    # Create temporary directories for intermediate files
    temp_dir = tempfile.mkdtemp()
    norm_output_dir = os.path.join(temp_dir, "normalized_files")
    mixed_output_dir = os.path.join(temp_dir, "mixed_output")

    try:
        # Normalize loudness
        logging.debug(f"Starting loudness normalization for {input_dir}.")
        normalize_loudness(input_dir, norm_output_dir)

        # Mix audio files
        logging.debug(f"Starting audio mixing for {norm_output_dir}.")
        mix_audio(norm_output_dir, mixed_output_dir)

        logging.info(f"Processed files are saved in: {mixed_output_dir}")
    finally:
        logging.debug(f"Temporary files are stored in {temp_dir}. Please inspect manually.")
        print(f"Temporary files are saved in: {temp_dir}")

if __name__ == "__main__":
    import argparse
    # Argument parser for input directory
    parser = argparse.ArgumentParser(description="Process audio files for normalization and mixing.")
    parser.add_argument("input_dir", help="The input directory containing audio files.")

    args = parser.parse_args()

    # Run the main function
    main(args.input_dir)
