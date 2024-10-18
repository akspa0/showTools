import argparse
import os
from pydub import AudioSegment

# Function to ensure output folder exists
def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Function to merge audio tracks
def merge_audio_tracks(input1, input2, output):
    # Load stereo tracks
    track1 = AudioSegment.from_file(input1)
    track2 = AudioSegment.from_file(input2)
    
    # Mix tracks
    mixed_track = track1.overlay(track2)

    # Ensure output folder exists
    output_folder = os.path.dirname(output)
    ensure_folder_exists(output_folder)
    
    # Export mixed track
    mixed_track.export(output, format="wav")

# Set up argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two audio tracks into one.")
    parser.add_argument("input1", help="Path to the first audio file")
    parser.add_argument("input2", help="Path to the second audio file")
    parser.add_argument("output", help="Path to the output audio file")

    args = parser.parse_args()

    merge_audio_tracks(args.input1, args.input2, args.output)
