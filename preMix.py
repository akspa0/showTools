import os
import argparse
from datetime import datetime
from pydub import AudioSegment

def adjust_channel_volumes_and_pan(audio):
    # Split stereo audio into left and right channels
    left_channel, right_channel = audio.split_to_mono()

    # Apply 20% panning to each channel for 40% stereo separation
    adjusted_left = left_channel.pan(-0.2)
    adjusted_right = right_channel.pan(0.2)

    # Combine left and right channels into stereo audio
    stereo_audio = adjusted_left.overlay(adjusted_right)

    return stereo_audio, adjusted_left, adjusted_right

def process_audio_files(input_folder, output_folder, append_audio_file=None, debug=False):
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the input directory
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            input_file = os.path.join(root, file)

            try:
                # Load stereo audio
                audio = AudioSegment.from_file(input_file)

                # Adjust volumes and pan for left and right channels
                stereo_audio, left_channel, right_channel = adjust_channel_volumes_and_pan(audio)

                # Export left and right channel files with original timestamp
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                timestamp = datetime.fromtimestamp(os.path.getmtime(input_file)).strftime('%Y-%m-%d_%H-%M-%S')
                left_output_path = os.path.join(output_folder, f"{base_name}_left_{timestamp}.wav")
                right_output_path = os.path.join(output_folder, f"{base_name}_right_{timestamp}.wav")
                left_channel.export(left_output_path, format='wav')
                right_channel.export(right_output_path, format='wav')

                # Rebuild stereo audio file
                stereo_output_path = os.path.join(output_folder, f"{base_name}_stereo_{timestamp}.wav")
                stereo_audio.export(stereo_output_path, format='wav')

                if append_audio_file:
                    # Append audio if provided
                    append_audio_segment = AudioSegment.from_file(append_audio_file)
                    stereo_audio_with_append = stereo_audio + append_audio_segment
                    stereo_output_path_with_append = os.path.join(output_folder, f"{base_name}_stereo_appended_{timestamp}.wav")
                    stereo_audio_with_append.export(stereo_output_path_with_append, format='wav')

                if debug:
                    print(f"Processed and saved: {input_file} -> {stereo_output_path}")

            except Exception as e:
                print(f"Error processing {input_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Process stereo audio files in a directory with 40% stereo separation, adjust peaks, and export separate channels and stereo output with original timestamps.')
    parser.add_argument('input', help='Input folder containing stereo audio files')
    parser.add_argument('output', help='Output folder where processed files will be saved')
    parser.add_argument('--append-audio', help='Audio file to append to the end of the resulting stereo output file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to show debug messages')

    args = parser.parse_args()

    input_folder = args.input
    output_folder = args.output
    append_audio_file = args.append_audio
    debug = args.debug

    process_audio_files(input_folder, output_folder, append_audio_file, debug)

if __name__ == "__main__":
    main()
