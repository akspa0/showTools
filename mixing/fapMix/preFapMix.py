import os
import re
import subprocess
import logging
import argparse
from datetime import datetime

# Set up logging
def setup_logging():
    log_filename = f"process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(filename=log_filename, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def create_subdirectories(output_dir):
    os.makedirs(os.path.join(output_dir, 'left'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'right'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'stereo'), exist_ok=True)

def run_command(command):
    try:
        logging.debug(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        logging.debug(f"Command output: {result.stdout}")
        return result
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with error: {e.stderr}")
        raise

def extract_timestamp(filename):
    match = re.search(r'(\d{10})\.\d{3}', filename)
    return match.group(1) if match else None

def process_files(input_dir, output_dir):
    create_subdirectories(output_dir)

    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    file_groups = {}

    # Group files by timestamp
    for file in files:
        timestamp = extract_timestamp(file)
        if timestamp:
            prefix = file.split('-')[0]
            suffix = file.split('_')[-1].split('.')[0]

            if prefix in ['trans_out', 'recv_out'] or suffix in ['in', 'out']:
                if timestamp not in file_groups:
                    file_groups[timestamp] = {'trans_out': None, 'recv_out': None, 'in': None, 'out': None}

                if prefix == 'trans_out':
                    file_groups[timestamp]['trans_out'] = file
                elif prefix == 'recv_out':
                    file_groups[timestamp]['recv_out'] = file
                elif suffix == 'in':
                    file_groups[timestamp]['in'] = file
                elif suffix == 'out':
                    file_groups[timestamp]['out'] = file

    # Process each group
    for timestamp, files in file_groups.items():
        file_paths = {key: os.path.join(input_dir, file) for key, file in files.items() if file}
        if not file_paths:
            continue
        
        # Extract channels
        for key in ['trans_out', 'recv_out']:
            if key in file_paths:
                channel_type = 'left' if key == 'trans_out' else 'right'
                output_path = os.path.join(output_dir, channel_type, f"{timestamp}.wav")
                run_command(['ffmpeg', '-i', file_paths[key], output_path])

        # Combine channels into stereo
        left_file = os.path.join(output_dir, 'left', f"{timestamp}.wav")
        right_file = os.path.join(output_dir, 'right', f"{timestamp}.wav")

        if os.path.exists(left_file) and os.path.exists(right_file):
            stereo_file = os.path.join(output_dir, 'stereo', f"{timestamp}_stereo.wav")
            run_command(['ffmpeg', '-i', left_file, '-i', right_file, '-filter_complex', 'amerge=inputs=2,pan=stereo|c0=c0|c1=c1', stereo_file])

def main():
    parser = argparse.ArgumentParser(description="Process audio files and generate separated and combined outputs.")
    parser.add_argument('input_dir', type=str, help='Path to the input directory containing audio files.')
    parser.add_argument('output_dir', type=str, help='Path to the output directory where processed files will be saved.')

    args = parser.parse_args()

    setup_logging()
    process_files(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
