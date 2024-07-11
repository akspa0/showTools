import os
import shutil
import argparse
from pydub import AudioSegment
from tqdm import tqdm

def calculate_peak_levels(audio):
    if audio.channels == 2:
        left_channel, right_channel = audio.split_to_mono()
        peaks = left_channel.max, right_channel.max
    else:
        peaks = audio.max, audio.max
    return peaks

def adjust_channel_volumes_and_pan(audio):
    left_channel, right_channel = audio.split_to_mono()
    left_peak, right_peak = calculate_peak_levels(audio)
    target_peak = max(left_peak, right_peak)

    # Calculate adjustment to match the lower peak to the higher one, considering headroom to avoid clipping
    adjustment = min(target_peak - left_peak, target_peak - right_peak, 0)

    adjusted_left = left_channel.apply_gain(adjustment)
    adjusted_right = right_channel.apply_gain(adjustment)

    # Pan channels for combined stereo output, not needed for individual channel exports
    panned_left = adjusted_left.pan(-0.2)
    panned_right = adjusted_right.pan(0.2)

    # Merge channels for stereo output
    stereo_audio = panned_left.overlay(panned_right)
    return stereo_audio, adjusted_left, adjusted_right

def export_channels(input_file, output_folder, stereo_audio, left_channel, right_channel, sample_rate=16000, timestamp=None, append_file=None):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    left_path = os.path.join(output_folder, "left", f"{base_name}_left.wav")
    right_path = os.path.join(output_folder, "right", f"{base_name}_right.wav")
    stereo_path = os.path.join(output_folder, "stereo", f"{base_name}_stereo.wav")

    # Export left and right channels with specified sample rate
    left_channel.export(left_path, format="wav", parameters=["-ar", str(sample_rate)])
    right_channel.export(right_path, format="wav", parameters=["-ar", str(sample_rate)])
    
    # Export stereo audio with specified sample rate
    stereo_audio.export(stereo_path, format="wav", parameters=["-ar", str(sample_rate)])

    # Set output file timestamps to match the input file timestamp
    if timestamp is not None:
        os.utime(left_path, (timestamp, timestamp))
        os.utime(right_path, (timestamp, timestamp))
        os.utime(stereo_path, (timestamp, timestamp))

    # Append the specified sound file to each output file if provided
    if append_file is not None:
        append_audio = AudioSegment.from_file(append_file)
        left_audio = AudioSegment.from_file(left_path)
        right_audio = AudioSegment.from_file(right_path)
        stereo_audio = AudioSegment.from_file(stereo_path)

        left_audio += append_audio
        right_audio += append_audio
        stereo_audio += append_audio

        left_audio.export(left_path, format="wav", parameters=["-ar", str(sample_rate)])
        right_audio.export(right_path, format="wav", parameters=["-ar", str(sample_rate)])
        stereo_audio.export(stereo_path, format="wav", parameters=["-ar", str(sample_rate)])

    # After appending and exporting, set the output file timestamps again to preserve the original timestamps
    if timestamp is not None:
        os.utime(left_path, (timestamp, timestamp))
        os.utime(right_path, (timestamp, timestamp))
        os.utime(stereo_path, (timestamp, timestamp))

def process_audio_file(input_file, output_folder, append_file=None, process_in_out=False, debug=False):
    try:
        # Get the duration of the input audio file
        audio = AudioSegment.from_file(input_file)
        duration_seconds = len(audio) / 1000  # Convert milliseconds to seconds

        # Skip processing if the duration is 8 seconds or less
        if duration_seconds <= 8:
            if debug:
                print(f"Skipped processing {input_file} as it is 8 seconds or less in duration.")
            return

        # Check if input file contains "_in" or "_out" in its filename
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        if process_in_out and ("_in" in base_name or "_out" in base_name):
            if "_out" in base_name:
                left_input = input_file
                right_input = input_file.replace("_out", "_in")
            else:  # "_in" in base_name
                left_input = input_file.replace("_in", "_out")
                right_input = input_file
            # Process the input files as left and right channels
            left_audio = AudioSegment.from_file(left_input)
            right_audio = AudioSegment.from_file(right_input)
        else:
            stereo_audio, left_audio, right_audio = adjust_channel_volumes_and_pan(audio)

        # Extract timestamp from input file
        timestamp = os.path.getmtime(input_file)

        # Create sub-folders if they don't exist
        os.makedirs(os.path.join(output_folder, "left"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "right"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "stereo"), exist_ok=True)

        # Export the processed audio channels and append the specified sound file
        export_channels(input_file, output_folder, stereo_audio, left_audio, right_audio, timestamp=timestamp, append_file=append_file)

        if debug:
            print(f"Processed and saved: {input_file} into separate channels and combined stereo.")

        # Move the output files to their respective sub-folders and set timestamps again
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        left_file = os.path.join(output_folder, f"left/{base_name}_left.wav")
        right_file = os.path.join(output_folder, f"right/{base_name}_right.wav")
        stereo_file = os.path.join(output_folder, f"stereo/{base_name}_stereo.wav")

        shutil.move(f"{base_name}_left.wav", left_file)
        shutil.move(f"{base_name}_right.wav", right_file)
        shutil.move(f"{base_name}_stereo.wav", stereo_file)

        os.utime(left_file, (timestamp, timestamp))
        os.utime(right_file, (timestamp, timestamp))
        os.utime(stereo_file, (timestamp, timestamp))

    except Exception as e:
        if debug:
            print(f"Error processing {input_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive audio processing script with individual channel exports, timestamp preservation, optional appending of a sound file, and processing input files with "_in" or "_out" in their filenames.')
    parser.add_argument('input', help='Input file or directory containing audio files')
    parser.add_argument('output', help='Output directory where processed files will be saved')
    parser.add_argument('--append', help='Optional file to append to the end of each output file')
    parser.add_argument('--process-in-out', action='store_true', help='Process input files with "_in" or "_out" in their filenames as left and right channels')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to show debug messages')

    args = parser.parse_args()

    input_path = args.input
    output_folder = args.output
    append_file = args.append
    process_in_out = args.process_in_out
    debug = args.debug

    if os.path.isfile(input_path):
        process_audio_file(input_path, output_folder, append_file, process_in_out, debug)
    elif os.path.isdir(input_path):
        for root, dirs, files in os.walk(input_path):
            for file in tqdm(files, desc="Processing files"):
                input_file = os.path.join(root, file)
                process_audio_file(input_file, output_folder, append_file, process_in_out, debug)

if __name__ == "__main__":
    main()
