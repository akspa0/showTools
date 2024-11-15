import os
import re
import logging
import argparse
from pydub import AudioSegment
from pydub.effects import normalize
import subprocess
from datetime import datetime
import shutil
import gc

logging.basicConfig(level=logging.INFO)

def apply_soft_limiter(audio, target_level=-6.0):
    peak = audio.max_dBFS
    if peak > target_level:
        gain_reduction = target_level - peak
        logging.info(f"Applying soft limiter with gain reduction of {gain_reduction} dB")
        return audio.apply_gain(gain_reduction)
    return audio

def loudness_normalization(audio):
    logging.info("Applying loudness normalization...")
    return normalize(audio)

def sanitize_text(text):
    sanitized = re.sub(r'[^\w\s]', '', text)
    return sanitized.replace(" ", "_")

def extract_first_words(lab_path, word_limit=12):
    with open(lab_path, 'r', encoding="utf-8") as lab_file:
        words = lab_file.read().split()[:word_limit]
    return sanitize_text(" ".join(words))

def run_command(command, description):
    try:
        subprocess.run(command, check=True)
        logging.info(f"{description} completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"{description} failed with error: {e}")
    except Exception as ex:
        logging.error(f"Unexpected error during {description}: {ex}")

def slicing(input_dir, output_dir):
    logging.info(f"Starting slicing for directory: {input_dir}")
    run_command(['fap', 'slice-audio-v2', '--no-merge-short', '--min-duration', '3', input_dir, output_dir], "Slicing")
    if not os.listdir(output_dir):
        logging.error("Slicing step completed, but no output files were created.")
        exit(1)
    logging.info(f"Slicing completed, files are available in {output_dir}")

def transcribe_directory(input_dir, num_workers=2):
    logging.info(f"Transcribing all files in directory: {input_dir} with {num_workers} workers")
    run_command(['fap', 'transcribe', '--lang', 'en', '--recursive', input_dir, '--num-workers', str(num_workers)], "Transcription")

def rename_and_copy_sliced_files(input_dir, target_dir, side):
    for root, dirs, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        audio_target_dir = os.path.join(target_dir, side, rel_path)
        os.makedirs(audio_target_dir, exist_ok=True)

        for filename in files:
            source_path = os.path.join(root, filename)

            if filename.endswith(".wav"):
                base_name = filename.split('.')[0]
                lab_filename = f"{base_name}.lab"
                lab_path = os.path.join(root, lab_filename)

                if os.path.exists(lab_path):
                    text_excerpt = extract_first_words(lab_path)
                    new_filename = f"{base_name}_{text_excerpt}.wav"
                else:
                    new_filename = filename

                target_path = os.path.join(audio_target_dir, new_filename)
                shutil.copy2(source_path, target_path)
                logging.info(f"Sliced audio file copied to {target_path}")

            elif filename.endswith(".lab"):
                target_path = os.path.join(audio_target_dir, filename)
                shutil.copy2(source_path, target_path)
                logging.info(f".lab file copied to {target_path}")

def process_audio_files(input_dir, output_dir, transcribe_left=False, transcribe_right=False, append_tones=False, normalize_audio=False, num_workers=2):
    logging.info("Starting audio processing")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    normalized_dir = os.path.join(run_output_dir, 'normalized')
    left_dir = os.path.join(run_output_dir, 'left')
    right_dir = os.path.join(run_output_dir, 'right')
    stereo_dir = os.path.join(run_output_dir, 'stereo')
    transcribed_and_sliced_dir = os.path.join(run_output_dir, 'transcribed-and-sliced')
    os.makedirs(normalized_dir, exist_ok=True)
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)
    os.makedirs(stereo_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        file_timestamp = os.path.getmtime(file_path)
        logging.info(f"Processing file: {filename}")
        
        try:
            audio = AudioSegment.from_file(file_path)
            limited_audio = apply_soft_limiter(audio)
            normalized_audio = loudness_normalization(limited_audio) if normalize_audio else limited_audio
            
            normalized_path = os.path.join(normalized_dir, filename)
            normalized_audio.export(normalized_path, format="wav")
            os.utime(normalized_path, (file_timestamp, file_timestamp))
            logging.info(f"Processed audio saved to {normalized_path}")

            del audio, limited_audio, normalized_audio
            gc.collect()

        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")
            continue

    channel_pairs = identify_channel_pairs(normalized_dir)
    process_channel_pairs(channel_pairs, left_dir, right_dir, stereo_dir, append_tones)

    if transcribe_left:
        slicing(left_dir, left_dir)
        transcribe_directory(left_dir, num_workers=num_workers)
        rename_and_copy_sliced_files(left_dir, transcribed_and_sliced_dir, 'left')

    if transcribe_right:
        slicing(right_dir, right_dir)
        transcribe_directory(right_dir, num_workers=num_workers)
        rename_and_copy_sliced_files(right_dir, transcribed_and_sliced_dir, 'right')

def identify_channel_pairs(normalized_dir):
    pairs = {}
    for filename in os.listdir(normalized_dir):
        base_key = filename.replace('trans_', '').replace('recv_', '')

        if filename.startswith('recv_'):
            pairs.setdefault(base_key, {})['left'] = os.path.join(normalized_dir, filename)
        elif filename.startswith('trans_'):
            pairs.setdefault(base_key, {})['right'] = os.path.join(normalized_dir, filename)

    return pairs


def process_channel_pairs(pairs, left_dir, right_dir, stereo_dir, append_tones):
    for base_name, channels in pairs.items():
        left_path = channels.get('left')
        right_path = channels.get('right')

        if left_path:
            save_channel_audio(left_path, left_dir)
        if right_path:
            save_channel_audio(right_path, right_dir)
        
        if left_path and right_path:
            mix_to_stereo(left_path, right_path, stereo_dir, base_name, append_tones)

def save_channel_audio(audio_path, target_dir):
    filename = os.path.basename(audio_path)
    output_path = os.path.join(target_dir, filename)
    AudioSegment.from_file(audio_path).export(output_path, format="wav")
    os.utime(output_path, (os.path.getmtime(audio_path), os.path.getmtime(audio_path)))
    logging.info(f"Channel audio saved to {output_path}")

def mix_to_stereo(left_path, right_path, stereo_dir, base_name, append_tones):
    try:
        left_audio = AudioSegment.from_file(left_path).pan(-0.2)
        right_audio = AudioSegment.from_file(right_path).pan(0.2)
        stereo_audio = left_audio.overlay(right_audio)

        if append_tones:
            tones = AudioSegment.from_file("tones.wav")
            stereo_audio += tones

        stereo_output_path = os.path.join(stereo_dir, f"{base_name}_stereo.wav")
        stereo_audio.export(stereo_output_path, format="wav")
        os.utime(stereo_output_path, (os.path.getmtime(left_path), os.path.getmtime(left_path)))
        logging.info(f"Stereo output saved to {stereo_output_path}")
    except Exception as e:
        logging.error(f"Error mixing stereo channels for {base_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files with soft limiting, optional normalization, and transcription.")
    parser.add_argument("--input-dir", required=True, help="Directory containing input audio files")
    parser.add_argument("--output-dir", required=True, help="Directory where output will be saved")
    parser.add_argument("--transcribe", action="store_true", help="Enable transcription for both left and right channels")
    parser.add_argument("--transcribe_left", action="store_true", help="Enable transcription for left channel only")
    parser.add_argument("--transcribe_right", action="store_true", help="Enable transcription for right channel only")
    parser.add_argument("--tones", action="store_true", help="Append 'tones.wav' to the end of stereo output files")
    parser.add_argument("--normalize", action="store_true", help="Enable loudness normalization")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of workers for transcription (default: 2)")

    args = parser.parse_args()
    
    transcribe_left = args.transcribe or args.transcribe_left
    transcribe_right = args.transcribe or args.transcribe_right

    process_audio_files(
        args.input_dir,
        args.output_dir,
        transcribe_left=transcribe_left,
        transcribe_right=transcribe_right,
        append_tones=args.tones,
        normalize_audio=args.normalize,
        num_workers=args.num_workers
    )
