import argparse
import os
from pydub import AudioSegment

def convert_to_mono(input_path, output_base_path):
    # Check if input path is a directory
    if os.path.isdir(input_path):
        for root, dirs, files in os.walk(input_path):
            for file in files:
                try:
                    input_file = os.path.join(root, file)
                    # Construct the output directory based on current root
                    output_dir = os.path.join(output_base_path, os.path.relpath(root, input_path))
                    # Create the output directory if it doesn't exist
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Construct the output file path
                    output_file = os.path.join(output_dir, file)
                    # Load the audio file
                    sound = AudioSegment.from_file(input_file)
                    # Convert to mono
                    mono_sound = sound.set_channels(1)
                    # Save the mono audio to output file
                    mono_sound.export(output_file, format=input_file.split('.')[-1])
                    print(f'Converted {input_file} to mono and saved as {output_file}')
                except Exception as e:
                    print(f'Failed to convert {input_file}: {e}')
    else:
        # Single file case, ensure output directory exists
        output_dir = os.path.dirname(output_base_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            # Load the audio file
            sound = AudioSegment.from_file(input_path)
            # Convert to mono
            mono_sound = sound.set_channels(1)
            # Save the mono audio to output file
            mono_sound.export(output_base_path, format=input_path.split('.')[-1])
            print(f'Converted {input_path} to mono and saved as {output_base_path}')
        except Exception as e:
            print(f'Failed to convert {input_path}: {e}')

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Convert audio files or all files in a directory (including subdirectories) to mono.")
    parser.add_argument("input_path", help="The input file or directory path")
    parser.add_argument("output_path", help="The output file or directory path")

    # Parse the arguments
    args = parser.parse_args()

    # Convert the input path to mono and save to the output path
    convert_to_mono(args.input_path, args.output_path)

if __name__ == "__main__":
    main()
