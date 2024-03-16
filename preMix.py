import os
import argparse
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
    # Create output folders if they don't exist
    left_folder = os.path.join(output_folder, "_left")
    right_folder = os.path.join(output_folder, "_right")
    stereo_folder = os.path.join(output_folder, "_stereo")
    appended_stereo_folder = os.path.join(output_folder, "_appended_stereo")
    os.makedirs(left_folder, exist_ok=True)
    os.makedirs(right_folder, exist_ok=True)
    os.makedirs(stereo_folder, exist_ok=True)
    os.makedirs(appended_stereo_folder, exist_ok=True)

    # Iterate over all files in the input directory
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            input_file = os.path.join(root, file)

            try:
                # Load stereo audio
                audio = AudioSegment.from_file(input_file)

                # Adjust volumes and pan for left and right channels
                stereo_audio, left_channel, right_channel = adjust_channel_volumes_and_pan(audio)

                # Get original timestamp of the input file
                timestamp = os.path.getmtime(input_file)

                # Export left and right channel files with original timestamp
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                left_output_path = os.path.join(left_folder, f"{base_name}_left.wav")
                right_output_path = os.path.join(right_folder, f"{base_name}_right.wav")
                left_channel.export(left_output_path, format='wav')
                right_channel.export(right_output_path, format='wav')

                # Rebuild stereo audio file with original timestamp
                stereo_output_path = os.path.join(stereo_folder, f"{base_name}_stereo.wav")
                stereo_audio.export(stereo_output_path, format='wav')

                if append_audio_file:
                    # Append audio if provided
                    append_audio_segment = AudioSegment.from_file(append_audio_file)
                    stereo_audio_with_append = stereo_audio + append_audio_segment
                    stereo_output_path_with_append = os.path.join(appended_stereo_folder, f"{base_name}_stereo_appended.wav")
                    stereo_audio_with_append.export(stereo_output_path_with_append, format='wav')

                if debug:
                    print(f"Processed: {input_file}")

                # Preserve original timestamp in output files
                os.utime(left_output_path, (timestamp, timestamp))
                os.utime(right_output_path, (timestamp, timestamp))
                os.utime(stereo_output_path, (timestamp, timestamp))
                if append_audio_file:
                    os.utime(stereo_output_path_with_append, (timestamp, timestamp))

            except Exception as e:
                print(f"Error processing {input_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Process stereo audio files in a directory with 40% stereo separation, adjust peaks, and export separate channels and stereo output.')
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
