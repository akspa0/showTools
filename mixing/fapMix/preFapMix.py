import os
import logging
from pydub import AudioSegment
from tqdm import tqdm
from datetime import datetime


def setup_logging(log_file):
    """Sets up logging for the script."""
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=log_file, level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')


def process_files(input_dir, output_dir):
    """Processes audio files with timestamp and stereo separation."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    left_dir = os.path.join(output_dir, 'left')
    right_dir = os.path.join(output_dir, 'right')
    stereo_dir = os.path.join(output_dir, 'stereo')
    for d in [left_dir, right_dir, stereo_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    file_groups = {}  # Dictionary to group files by timestamp

    for file in files:
        if file.lower().endswith('.wav') and os.path.getsize(os.path.join(input_dir, file)) >= 10240:  # Ignore files less than 10KB
            timestamp = file.split('-')[-1].split('.')[0]  # Extract timestamp
            if timestamp not in file_groups:
                file_groups[timestamp] = []
            file_groups[timestamp].append(file)

    for timestamp, files in tqdm(file_groups.items(), desc="Processing files"):
        left_files = [f for f in files if f.startswith('recv_out-')]
        right_files = [f for f in files if f.startswith('trans_out-')]

        # Process left channel with timestamp in output filename
        for left_file in left_files:
            input_path = os.path.join(input_dir, left_file)
            output_path = os.path.join(left_dir, f'{timestamp}_{left_file}')
            audio = AudioSegment.from_wav(input_path)
            audio.set_frame_rate(16000).set_sample_width(2).export(output_path, format="wav")
            logging.info(f'Processed left channel: {output_path}')

        # Process right channel with timestamp in output filename
        for right_file in right_files:
            input_path = os.path.join(input_dir, right_file)
            output_path = os.path.join(right_dir, f'{timestamp}_{right_file}')
            audio = AudioSegment.from_wav(input_path)
            audio.set_frame_rate(16000).set_sample_width(2).export(output_path, format="wav")
            logging.info(f'Processed right channel: {output_path}')

        # Process stereo output with timestamp and adjust separation using pydub
        if left_files and right_files:
            left_input = os.path.join(left_dir, f'{timestamp}_{left_files[0]}')
            right_input = os.path.join(right_dir, f'{timestamp}_{right_files[0]}')
            combined_output = os.path.join(stereo_dir, f'combined-{left_files[0]}.wav')  # Use combined filename

            left_audio = AudioSegment.from_wav(left_input)
            right_audio = AudioSegment.from_wav(right_input)

            # Pan and merge using pydub
            stereo_audio = pan_and_merge(left_audio, right_audio)
            stereo_audio.export(combined_output, format="wav")
            logging.info(f'Processed stereo audio: {combined_output}')


def pan_and_merge(left_channel, right_channel):
    """
    Applies panning to left and right channels and merges them into a stereo track.
    Left channel is panned slightly left, right channel is panned slightly right.
    """
    # Pan channels
    panned_left = left_channel.pan(-0.2)
    panned_right = right_channel.pan(0.2)

    # Merge channels for stereo output
    stereo_audio = panned_left.overlay(panned_right)
    return stereo_audio


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process audio files.')
    parser.add_argument('input_dir', type=str, help='Directory containing input audio files.')
    parser.add_argument('output_dir', type=str, help='Directory to store output files.')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.output_dir, f'{timestamp}_processing_log.txt')
    setup_logging(log_file)

    process_files(args.input_dir, args.output_dir)
