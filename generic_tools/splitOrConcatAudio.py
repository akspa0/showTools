import argparse
from pydub import AudioSegment
import os
import audioread

def split_audio(input_file, output_folder, segment_length_ms=15*60*1000):
    # Load the input audio file
    audio = AudioSegment.from_file(input_file)

    # Calculate the number of segments
    num_segments = len(audio) // segment_length_ms + 1

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Split the audio into segments of specified length
    for i in range(num_segments):
        start_time = i * segment_length_ms
        end_time = min((i + 1) * segment_length_ms, len(audio))
        segment = audio[start_time:end_time]
        segment.export(os.path.join(output_folder, f"segment_{i+1}.{input_file.split('.')[-1]}"), format=input_file.split('.')[-1])

    print(f"{num_segments} audio segments created successfully.")

def concatenate_segments(input_folder, output_file):
    print("Input folder:", input_folder)
    print("Output file:", output_file)

    # Get a list of all audio segments in the input folder
    segment_files = [
        file for file in os.listdir(input_folder)
        if any(char.isdigit() for char in file) and file.endswith(('.wav', '.mp3', '.flac', '.ogg'))  # Include files with numbers and specified formats
    ]
    segment_files.sort()  # Ensure segments are concatenated in order

    print("Input segment files:", segment_files)

    # Initialize the combined audio segment
    combined_audio = AudioSegment.empty()
    secondary_segments = []

    # Group segments with different suffixes for secondary concatenation
    grouped_segments = {}
    for file in segment_files:
        prefix, suffix = extract_prefix_suffix(file)
        if prefix not in grouped_segments:
            grouped_segments[prefix] = []
        grouped_segments[prefix].append((file, suffix))

    # Concatenate primary segments and collect secondary segments
    for prefix, files in grouped_segments.items():
        primary_segments = [file for file, suffix in files if all(suffix == other_suffix for _, other_suffix in files)]
        secondary_segments += [file for file, suffix in files if any(suffix != other_suffix for _, other_suffix in files)]
        
        # Concatenate primary segments
        for file in primary_segments:
            file_path = os.path.join(input_folder, file)
            print("Processing primary file:", file_path)
            segment = AudioSegment.from_file(file_path)
            print("Segment duration:", len(segment), "ms")
            combined_audio += segment

    # Concatenate secondary segments
    for file in secondary_segments:
        file_path = os.path.join(input_folder, file)
        print("Processing secondary file:", file_path)
        segment = AudioSegment.from_file(file_path)
        print("Segment duration:", len(segment), "ms")
        combined_audio += segment

    # Extract the output file extension
    output_extension = output_file.split('.')[-1]

    # Export the combined audio to the output file using the determined format
    combined_audio.export(output_file, format=output_extension)

    print("Audio segments concatenated successfully.")

def extract_prefix_suffix(file_name):
    # Extract prefix and suffix numbers from the filename
    parts = file_name.split('_')
    prefix = parts[0] if parts[0].isdigit() else ''
    suffix = parts[-1] if parts[-1].isdigit() else ''
    return prefix, suffix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split and concatenate audio files.")
    parser.add_argument("--input_file", help="Path to the input audio file to split.")
    parser.add_argument("--output_folder", help="Path to the output folder for segments.")
    parser.add_argument("--segment_length", type=int, default=900, help="Segment length in seconds (default: 900).")
    parser.add_argument("--operation", choices=["split", "concat"], help="Specify 'split' to split the audio file or 'concat' to concatenate segments.")
    parser.add_argument("--input_folder", help="Path to the input folder containing segments (for concatenation operation).")
    parser.add_argument("--output_file", help="Path to the output file for concatenated audio (for concatenation operation).")

    args = parser.parse_args()

    if args.operation == "split":
        split_audio(args.input_file, args.output_folder, args.segment_length * 1000)  # Convert seconds to milliseconds
    elif args.operation == "concat":
        concatenate_segments(args.input_folder, args.output_file)
