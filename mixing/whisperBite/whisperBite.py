import os
import sys
import argparse
from datetime import datetime
import logging
from whisper import load_model
from pyannote.audio import Pipeline
import torch
from utils import sanitize_filename, download_audio, zip_results, convert_and_normalize_audio, default_slicing, normalize_audio_ffmpeg
from vocal_separation import separate_vocals_with_demucs

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def slice_audio_by_speaker(file_path, diarization, speaker_output_dir):
    """Slice audio by speakers based on diarization results."""
    from pydub import AudioSegment

    audio = AudioSegment.from_file(file_path)
    os.makedirs(speaker_output_dir, exist_ok=True)

    speaker_files = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_dir = os.path.join(speaker_output_dir, f"Speaker_{speaker}")
        os.makedirs(speaker_dir, exist_ok=True)

        segment_audio = audio[turn.start * 1000:turn.end * 1000]
        segment_path = os.path.join(speaker_dir, f"segment_{int(turn.start)}_{int(turn.end)}.wav")
        segment_audio.export(segment_path, format="wav")
        logging.info(f"Saved segment: {segment_path}")

        if speaker not in speaker_files:
            speaker_files[speaker] = []
        speaker_files[speaker].append(segment_path)

    return speaker_files

def transcribe_with_whisper(model, audio_files, output_dir):
    """Transcribe audio files using Whisper."""
    for audio_file in audio_files:
        logging.info(f"Transcribing file: {audio_file}")
        transcription = model.transcribe(audio_file)
        transcription_text = transcription['text'].strip()

        base_name = sanitize_filename(transcription_text[:50] if transcription_text else "transcription")
        transcription_file = os.path.join(output_dir, f"{base_name}.txt")
        audio_output_file = os.path.join(output_dir, f"{base_name}.wav")

        with open(transcription_file, "w") as f:
            f.write(transcription_text)

        os.rename(audio_file, audio_output_file)
        logging.info(f"Transcription saved to {transcription_file} and renamed audio to {audio_output_file}")

def process_audio(input_path, output_dir, model_name, enable_vocal_separation, num_speakers):
    """Unified pipeline for audio processing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Expected number of speakers: {num_speakers}")

    processed_files = []
    audio_files = [input_path] if os.path.isfile(input_path) else [
        os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(('.mp3', '.wav', '.m4a'))
    ]

    for audio_file in audio_files:
        logging.info(f"Processing file: {audio_file}")
        base_output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(audio_file))[0])
        os.makedirs(base_output_dir, exist_ok=True)

        try:
            normalized_file = convert_and_normalize_audio(audio_file, base_output_dir)
            if enable_vocal_separation:
                logging.info("Performing vocal separation.")
                vocals_file = separate_vocals_with_demucs(normalized_file, base_output_dir)
                if os.path.exists(vocals_file):
                    processed_files.append((vocals_file, base_output_dir))
                else:
                    logging.error(f"Vocal-separated file {vocals_file} not found.")
                    continue
            else:
                processed_files.append((normalized_file, base_output_dir))
        except Exception as e:
            logging.error(f"Error processing {audio_file}: {e}")
            continue

    model = load_model(model_name)
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    pipeline.to(device)

    for processed_file, file_output_dir in processed_files:
        try:
            speaker_output_dir = os.path.join(file_output_dir, "speakers")
            diarization = pipeline(processed_file)
            speaker_files = slice_audio_by_speaker(processed_file, diarization, speaker_output_dir)

            for speaker, segments in speaker_files.items():
                speaker_transcription_dir = os.path.join(file_output_dir, f"Speaker_{speaker}_transcriptions")
                os.makedirs(speaker_transcription_dir, exist_ok=True)

                # Normalize and boost soundbites after slicing
                normalized_segments = []
                for segment in segments:
                    normalized_segment = normalize_audio_ffmpeg(segment)
                    normalized_segments.append(normalized_segment)

                transcribe_with_whisper(model, normalized_segments, speaker_transcription_dir)
        except Exception as e:
            logging.error(f"Error during diarization or transcription for {processed_file}: {e}")
            continue

        # Zip the results for the processed file
        zip_results(file_output_dir, processed_file)

def main():
    parser = argparse.ArgumentParser(
        description="Streamlined audio processing script for transcription and diarization."
    )
    parser.add_argument('--input_dir', type=str, help='Directory containing input audio files.')
    parser.add_argument('--input_file', type=str, help='Single audio file for processing.')
    parser.add_argument('--url', type=str, help='URL to download audio from.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files.')
    parser.add_argument('--model', type=str, default="turbo", help='Whisper model to use (default: turbo).')
    parser.add_argument('--num_speakers', type=int, default=2, help='Number of speakers for diarization (default: 2).')
    parser.add_argument('--enable_vocal_separation', action='store_true', help='Enable vocal separation using Demucs.')
    args = parser.parse_args()

    if not any([args.input_dir, args.input_file, args.url]):
        parser.print_help()
        sys.exit(1)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    main_output_dir = os.path.join(args.output_dir, f"output_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)

    if args.url:
        input_path = download_audio(args.url, main_output_dir)
    else:
        input_path = args.input_dir if args.input_dir else args.input_file

    if not input_path or not os.path.exists(input_path):
        logging.error("No valid input path provided.")
        parser.print_help()
        sys.exit(1)

    process_audio(
        input_path,
        main_output_dir,
        args.model,
        args.enable_vocal_separation,
        args.num_speakers
    )

if __name__ == "__main__":
    main()
