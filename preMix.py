import os
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

def export_channels(input_file, output_folder, stereo_audio, left_channel, right_channel):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    left_path = os.path.join(output_folder, f"{base_name}_left.wav")
    right_path = os.path.join(output_folder, f"{base_name}_right.wav")
    stereo_path = os.path.join(output_folder, f"{base_name}_stereo.wav")

    left_channel.export(left_path, format="wav")
    right_channel.export(right_path, format="wav")
    stereo_audio.export(stereo_path, format="wav")

def process_audio_file(input_file, output_folder, debug=False):
    try:
        audio = AudioSegment.from_file(input_file)
        stereo_audio, left_channel, right_channel = adjust_channel_volumes_and_pan(audio)

        # Create the output directory if it does not exist
        os.makedirs(output_folder, exist_ok=True)

        # Export the processed audio channels
        export_channels(input_file, output_folder, stereo_audio, left_channel, right_channel)

        if debug:
            print(f"Processed and saved: {input_file} into separate channels and combined stereo.")

    except Exception as e:
        if debug:
            print(f"Error processing {input_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive audio processing script with individual channel exports.')
    parser.add_argument('input', help='Input file or directory containing audio files')
    parser.add_argument('output', help='Output directory where processed files will be saved')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to show debug messages')

    args = parser.parse_args()

    input_path = args.input
    output_folder = args.output
    debug = args.debug

    if os.path.isfile(input_path):
        process_audio_file(input_path, output_folder, debug)
    elif os.path.isdir(input_path):
        for root, dirs, files in os.walk(input_path):
            for file in tqdm(files, desc="Processing files"):
                input_file = os.path.join(root, file)
                process_audio_file(input_file, output_folder, debug)

if __name__ == "__main__":
    main()
