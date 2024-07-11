import os
import argparse
from datetime import datetime
from pydub import AudioSegment
from tqdm import tqdm
import numpy as np

def calculate_noise_floor(audio):
    # Calculate the noise floor as the average level of background noise
    # This can be done by analyzing a portion of silent audio
    # For simplicity, we'll calculate it using the first 500 milliseconds of the audio
    silent_duration = 500  # milliseconds
    silent_audio = audio[:silent_duration]
    noise_floor = silent_audio.dBFS
    return noise_floor

def compress_audio(audio, noise_floor):
    # Adjust compression parameters as needed
    threshold = noise_floor  # Set the threshold to the noise floor
    ratio = 2.0  # Compression ratio
    attack = 10  # Attack time in milliseconds
    release = 100  # Release time in milliseconds

    # Apply dynamic range compression
    compressed_audio = audio.compress_dynamic_range(threshold, ratio, attack, release)
    return compressed_audio

def merge_mono_to_stereo(left_channel, right_channel):
    # Offset each channel by 20% from the center
    if len(left_channel) > 0 and len(right_channel) > 0:  # Ensure non-empty channels
        center = len(left_channel) // 2
        offset = int(center * 0.2)
        left_channel = left_channel.pan(-0.2)
        right_channel = right_channel.pan(0.2)
        return left_channel.overlay(right_channel)
    else:
        return None

def append_audio(input_audio, append_audio_segment):
    # Ensure sample rate matches
    if input_audio.frame_rate != append_audio_segment.frame_rate:
        append_audio_segment = append_audio_segment.set_frame_rate(input_audio.frame_rate)
    return input_audio.append(append_audio_segment, crossfade=0)

def normalize_audio(audio):
    # Normalize each channel separately
    normalized_channels = [channel.normalize() for channel in audio.split_to_mono()]
    # Merge normalized channels back into stereo audio
    normalized_audio = AudioSegment.from_mono_audiosegments(*normalized_channels)
    return normalized_audio

def process_audio_file(input_file, output_folder, append_audio_segment=None, debug=False, pre_split=False):
    try:
        if pre_split:
            left_channel = AudioSegment.from_file(input_file[0])
            right_channel = AudioSegment.from_file(input_file[1])
        else:
            sound = AudioSegment.from_file(input_file)
            left_channel = sound.split_to_mono()[0]
            right_channel = sound.split_to_mono()[1]

        # Calculate noise floor for each channel
        noise_floor_left = calculate_noise_floor(left_channel)
        noise_floor_right = calculate_noise_floor(right_channel)

        # Compress each channel using noise floor
        compressed_left_channel = compress_audio(left_channel, noise_floor_left)
        compressed_right_channel = compress_audio(right_channel, noise_floor_right)

        # Normalize each channel
        normalized_left_channel = normalize_audio(compressed_left_channel)
        normalized_right_channel = normalize_audio(compressed_right_channel)

        # Merge mono channels to stereo with offset
        merged_audio = merge_mono_to_stereo(normalized_left_channel, normalized_right_channel)

        if merged_audio is not None:
            # Append audio if provided
            if append_audio_segment:
                merged_audio_with_append = append_audio(merged_audio, append_audio_segment)
            else:
                merged_audio_with_append = merged_audio

            # Get last modification time of input file
            last_modified = datetime.fromtimestamp(os.path.getmtime(input_file))

            # Create output folder if it doesn't exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Construct output filename
            output_filename = os.path.join(output_folder, os.path.splitext(os.path.basename(input_file))[0] + ".wav")

            # Save merged audio with same last modification time as the input file
            merged_audio_with_append.export(output_filename, format='wav', parameters=['-ac', '2', '-ar', '44100', '-sample_fmt', 's16', '-filter:a', 'loudnorm'])
            os.utime(output_filename, (last_modified.timestamp(), last_modified.timestamp()))

            # Print debug text if enabled
            if debug:
                print(f"Processed: {input_file} -> {output_filename}")

            # Create a folder for split audio
            split_output_folder = os.path.join(output_folder, last_modified.strftime('%Y-%m-%d') + "-split")
            if not os.path.exists(split_output_folder):
                os.makedirs(split_output_folder)

            # Save Left and Right audio files separately with original time and date
            left_output_file = os.path.join(split_output_folder, os.path.basename(input_file).replace(".wav", f"_left.wav"))
            right_output_file = os.path.join(split_output_folder, os.path.basename(input_file).replace(".wav", f"_right.wav"))
            normalized_left_channel.export(left_output_file, format='wav', parameters=['-ac', '1', '-ar', '44100', '-sample_fmt', 's16', '-filter:a', 'loudnorm'])
            normalized_right_channel.export(right_output_file, format='wav', parameters=['-ac', '1', '-ar', '44100', '-sample_fmt', 's16', '-filter:a', 'loudnorm'])
        else:
            print(f"Skipping {input_file}: Empty channels")

    except Exception as e:
        print(f"Error processing {input_file}: {e}")

def process_audio_files(input_path, output_folder, append_audio_segment=None, debug=False, pre_split=False):
    if os.path.isfile(input_path):
        process_audio_file(input_path, output_folder, append_audio_segment, debug, pre_split)
    elif os.path.isdir(input_path):
        for root, dirs, files in os.walk(input_path):
            for file in tqdm(files, desc="Processing files"):
                input_file = os.path.join(root, file)
                process_audio_file(input_file, output_folder, append_audio_segment, debug, pre_split)

def main():
    parser = argparse.ArgumentParser(description='Process stereo audio files, split them into two mono streams, normalize and compress the audio slightly, then merge the two streams into a single output file.')
    parser.add_argument('input', help='Input file or directory containing audio files')
    parser.add_argument('output', help='Output directory where processed files will be saved')
    parser.add_argument('--append-audio', help='Audio file to append to the end of each input file')
    parser.add_argument('--pre-split', action='store_true', help='Treat input files as pre-split left and right channels')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to show debug messages')

    args = parser.parse_args()

    input_path = args.input
    output_folder = args.output
    append_audio_file = args.append_audio
    pre_split = args.pre_split
    debug = args.debug

    if append_audio_file:
        # Load the append audio file
        append_audio_segment = AudioSegment.from_file(append_audio_file)
    else:
        append_audio_segment = None

    process_audio_files(input_path, output_folder, append_audio_segment, debug, pre_split)

    # Append audio to all processed files
    if append_audio_segment:
        append_audio_to_all_files(output_folder, append_audio_segment, debug)

if __name__ == "__main__":
    main()
