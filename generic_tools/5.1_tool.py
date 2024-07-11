import os
import argparse
from pydub import AudioSegment
import numpy as np
import wave
from tqdm import tqdm

def split_5_1_channels(input_file, output_dir):
    audio = AudioSegment.from_file(input_file, format="flac")
    
    # Resample to 48kHz
    audio = audio.set_frame_rate(48000)
    
    samples = np.array(audio.get_array_of_samples())
    
    assert audio.channels == 6, "Input audio must have 6 channels (5.1 surround sound)"
    
    samples = samples.reshape((-1, audio.channels))
    
    channels = [samples[:, i] for i in range(audio.channels)]
    channel_names = ["FC", "FL", "FR", "BL", "BR", "LFE"]
    
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    output_subdir = os.path.join(output_dir, base_filename)
    os.makedirs(output_subdir, exist_ok=True)
    
    for i, channel in enumerate(channels):
        output_file = os.path.join(output_subdir, f"{channel_names[i]}.wav")
        with wave.open(output_file, "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(4)  # 32-bit samples
            wav_file.setframerate(48000)
            wav_file.writeframes(channel.tobytes())

def process_directory(input_dir, output_dir):
    files = [f for f in os.listdir(input_dir) if f.endswith(".flac")]
    for file in tqdm(files, desc="Processing directory", unit="file"):
        input_file = os.path.join(input_dir, file)
        split_5_1_channels(input_file, output_dir)

def combine_channels_to_5_1(input_dir, output_file):
    channel_names = ["FC", "FL", "FR", "BL", "BR", "LFE"]
    channels = []
    
    for channel_name in channel_names:
        input_channel_file = os.path.join(input_dir, f"{channel_name}.wav")
        audio = AudioSegment.from_file(input_channel_file, format="wav")
        channels.append(np.array(audio.get_array_of_samples()))

    combined_samples = np.stack(channels, axis=-1).reshape(-1)
    combined_audio = AudioSegment(
        combined_samples.tobytes(), 
        frame_rate=48000, 
        sample_width=4, 
        channels=6
    )

    combined_audio.export(output_file, format="flac")

def main():
    parser = argparse.ArgumentParser(description="Split and combine 5.1 channel FLAC audio files.")
    parser.add_argument("mode", choices=["split", "combine"], help="Mode: 'split' to split channels, 'combine' to combine channels.")
    parser.add_argument("input", type=str, help="Input file or directory")
    parser.add_argument("output", type=str, help="Output directory or file")
    
    args = parser.parse_args()
    
    if args.mode == "split":
        if os.path.isfile(args.input):
            split_5_1_channels(args.input, args.output)
        elif os.path.isdir(args.input):
            process_directory(args.input, args.output)
        else:
            raise ValueError("Input must be a file or a directory")
    elif args.mode == "combine":
        if os.path.isdir(args.input):
            combine_channels_to_5_1(args.input, args.output)
        else:
            raise ValueError("Input must be a directory containing the channel files")

if __name__ == "__main__":
    main()
