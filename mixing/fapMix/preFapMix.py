import os
import re
import subprocess

def get_unix_timestamp(filename):
    match = re.search(r'(\d{10})(?:\.\d+)?', filename)
    if match:
        return match.group(1)
    return None

def process_files(input_dir):
    files = os.listdir(input_dir)
    
    # Filters to identify relevant files
    file_filters = (r'trans_out-', r'recv_out-', r'_in', r'_out')
    
    # Group files by timestamp
    file_groups = {}
    
    for file in files:
        if any(file.startswith(prefix) or suffix in file for prefix, suffix in zip(file_filters, file_filters)):
            timestamp = get_unix_timestamp(file)
            if timestamp:
                if timestamp not in file_groups:
                    file_groups[timestamp] = []
                file_groups[timestamp].append(file)
    
    for timestamp, file_list in file_groups.items():
        if len(file_list) < 2:
            print(f"Not enough files to process for timestamp {timestamp}")
            continue
        
        # Paths for files
        trans_files = [f for f in file_list if f.startswith('trans_out-')]
        recv_files = [f for f in file_list if f.startswith('recv_out-')]
        in_files = [f for f in file_list if '_in' in f]
        out_files = [f for f in file_list if '_out' in f]
        
        if len(trans_files) > 0 and len(recv_files) > 0:
            # Assume processing both trans_out and recv_out files
            process_channels(input_dir, timestamp, trans_files, recv_files)
        elif len(in_files) > 0 and len(out_files) > 0:
            # Assume processing in and out files
            process_channels(input_dir, timestamp, in_files, out_files)
        else:
            print(f"File grouping for timestamp {timestamp} is incomplete.")

def process_channels(input_dir, timestamp, in_files, out_files):
    # Implement the logic to mix channels
    # Paths for mixed outputs
    mixed_output_dir = os.path.join(input_dir, 'mixed_output')
    os.makedirs(mixed_output_dir, exist_ok=True)
    
    # Example processing logic
    for in_file in in_files:
        for out_file in out_files:
            # Process and mix files here
            # Example ffmpeg command (adjust based on actual needs)
            in_file_path = os.path.join(input_dir, in_file)
            out_file_path = os.path.join(input_dir, out_file)
            output_file_path = os.path.join(mixed_output_dir, f'{timestamp}_mixed.wav')
            
            command = [
                'ffmpeg', '-i', in_file_path, '-i', out_file_path,
                '-filter_complex', '[0:a][1:a]amerge=inputs=2[a]',
                '-map', '[a]', '-ac', '2', output_file_path
            ]
            
            print(f"Running command: {' '.join(command)}")
            try:
                subprocess.run(command, check=True, text=True, capture_output=True)
                print(f"Processed and mixed: {output_file_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing files: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process audio files")
    parser.add_argument("input_dir", help="Input directory containing audio files")
    args = parser.parse_args()
    
    process_files(args.input_dir)
