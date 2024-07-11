import os
import argparse
import random
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm

def extract_random_samples(input_folder, output_folder, duration_seconds):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Initialize combined output audio
    combined_audio = AudioSegment.empty()

    # List to track if we've included a sample from any file
    included_from_file = False

    # Iterate through each input file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav") or filename.endswith(".mp3"):  # Adjust file extensions as needed
            file_path = os.path.join(input_folder, filename)

            # Load the audio file
            audio = AudioSegment.from_file(file_path)

            # Split audio based on silence (VAD)
            audio_chunks = split_on_silence(
                audio,
                min_silence_len=500,  # Adjust as needed for VAD sensitivity
                silence_thresh=-40  # Adjust as needed for VAD sensitivity
            )

            # Randomly select a chunk of specified duration from each file
            random_chunk = random.choice(audio_chunks)
            while is_introduction(random_chunk) or (not included_from_file and len(audio_chunks) == 1):
                random_chunk = random.choice(audio_chunks)

            if len(random_chunk) >= duration_seconds * 1000:
                start_time = random.randint(0, len(random_chunk) - duration_seconds * 1000)
                sample = random_chunk[start_time:start_time + duration_seconds * 1000]

                # Append sample to combined output audio
                combined_audio += sample

                included_from_file = True  # Set to True after including a sample from any file

    # Export combined output audio to output folder
    output_filename = f"combined_sample_{duration_seconds}s.wav"
    output_path = os.path.join(output_folder, output_filename)
    combined_audio.export(output_path, format="wav")

def is_introduction(audio_chunk, threshold=4000):
    # Check if the audio chunk is likely to be an introduction (e.g., short duration)
    return len(audio_chunk) < threshold

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract random samples from input audio files and combine them into output samples.")
    parser.add_argument("input_folder", help="Path to the folder containing input audio files.")
    parser.add_argument("output_folder", help="Path to the folder where output audio samples will be saved.")
    parser.add_argument("--15", action="store_true", help="Generate output sample of 15 seconds.")
    parser.add_argument("--30", action="store_true", help="Generate output sample of 30 seconds.")
    parser.add_argument("--45", action="store_true", help="Generate output sample of 45 seconds.")
    parser.add_argument("--60", action="store_true", help="Generate output sample of 60 seconds.")
    
    args = parser.parse_args()

    if args.__dict__["15"]:
        tqdm.write("Processing audio samples...")
        extract_random_samples(args.input_folder, args.output_folder, 15)
        tqdm.write("Audio processing completed.")
    elif args.__dict__["30"]:
        tqdm.write("Processing audio samples...")
        extract_random_samples(args.input_folder, args.output_folder, 30)
        tqdm.write("Audio processing completed.")
    elif args.__dict__["45"]:
        tqdm.write("Processing audio samples...")
        extract_random_samples(args.input_folder, args.output_folder, 45)
        tqdm.write("Audio processing completed.")
    elif args.__dict__["60"]:
        tqdm.write("Processing audio samples...")
        extract_random_samples(args.input_folder, args.output_folder, 60)
        tqdm.write("Audio processing completed.")
    else:
        print("Please specify the length of the output sample using --15, --30, --45, or --60.")
