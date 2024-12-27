import os
import sys
import argparse
from datetime import datetime
import logging
from pydub import AudioSegment
from whisper import load_model
from pyannote.audio import Pipeline
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def sanitize_filename(name, max_length=128):
    """Sanitize filename by removing unwanted characters and limiting length."""
    sanitized = "".join(char if char.isalnum() or char == "_" else "_" for char in name)[:max_length]
    return sanitized

def convert_and_normalize_audio(input_audio, output_dir):
    """Convert audio to WAV with 44.1kHz, 16-bit and normalize loudness to -14 LUFS."""
    try:
        logging.info(f"Processing audio file: {input_audio}")
        normalized_dir = os.path.join(output_dir, "normalized")
        os.makedirs(normalized_dir, exist_ok=True)

        audio = AudioSegment.from_file(input_audio)

        # Set sample rate and bit depth
        audio = audio.set_frame_rate(44100).set_sample_width(2)

        # Normalize loudness to -14 LUFS
        change_in_dBFS = -14 - audio.dBFS
        normalized_audio = audio.apply_gain(change_in_dBFS)

        # Export normalized audio
        base_name = os.path.splitext(os.path.basename(input_audio))[0]
        output_file = os.path.join(normalized_dir, f"{base_name}_normalized.wav")
        normalized_audio.export(output_file, format="wav")
        logging.info(f"Normalized audio saved to {output_file}")
        return output_file
    except Exception as e:
        logging.error(f"Error normalizing {input_audio}: {e}")
        raise

def separate_vocals_with_demucs(input_audio, output_dir):
    """Separate vocals from an audio file using Demucs."""
    demucs_output_dir = os.path.join(output_dir, "demucs")
    os.makedirs(demucs_output_dir, exist_ok=True)

    try:
        subprocess.run(
            [
                "demucs",
                "--two-stems", "vocals",
                "--out", demucs_output_dir,
                input_audio
            ],
            check=True
        )
        vocals_file = os.path.join(demucs_output_dir, "vocals.wav")
        logging.info(f"Vocal separation completed. File saved to {vocals_file}")
        return vocals_file
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during vocal separation: {e}")
        raise

def slice_audio_by_speaker(file_path, diarization, speaker_output_dir):
    """Slice audio by speakers based on diarization results."""
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

def process_audio(input_path, output_dir, model_name, num_speakers, enable_vocal_separation):
    """Unified pipeline for audio processing."""
    processed_files = []
    audio_files = [input_path] if os.path.isfile(input_path) else [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(('.mp3', '.wav', '.m4a'))]

    for audio_file in audio_files:
        try:
            logging.info(f"Starting normalization for {audio_file}")
            normalized_file = convert_and_normalize_audio(audio_file, output_dir)
            logging.info(f"Normalization completed for {normalized_file}")

            if enable_vocal_separation:
                logging.info(f"Starting vocal separation for {normalized_file}")
                vocals_file = separate_vocals_with_demucs(normalized_file, output_dir)
                processed_files.append(vocals_file)
                logging.info(f"Vocal separation completed for {vocals_file}")
            else:
                processed_files.append(normalized_file)
        except Exception as e:
            logging.error(f"Error processing {audio_file}: {e}")

    logging.info("Audio processing completed.")

    model = load_model(model_name)
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

    for processed_file in processed_files:
        logging.info(f"Starting diarization for {processed_file}")
        speaker_output_dir = os.path.join(output_dir, "speakers")
        diarization = pipeline(processed_file)
        logging.info(f"Diarization completed for {processed_file}")

        speaker_files = slice_audio_by_speaker(processed_file, diarization, speaker_output_dir)

        for speaker, segments in speaker_files.items():
            speaker_transcription_dir = os.path.join(output_dir, f"Speaker_{speaker}_transcriptions")
            os.makedirs(speaker_transcription_dir, exist_ok=True)
            logging.info(f"Starting transcription for Speaker {speaker}")
            transcribe_with_whisper(model, segments, speaker_transcription_dir)
            logging.info(f"Transcription completed for Speaker {speaker}")

def main():
    parser = argparse.ArgumentParser(description="Streamlined audio processing script.")
    parser.add_argument('--input_dir', type=str, help='Directory containing input audio files')
    parser.add_argument('--input_file', type=str, help='Single audio file for processing')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files')
    parser.add_argument('--model', type=str, default="turbo", help='Whisper model to use (default: turbo)')
    parser.add_argument('--num_speakers', type=int, default=2, help='Expected number of speakers in the audio')
    parser.add_argument('--enable_vocal_separation', action='store_true', help='Enable vocal separation using Demucs')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    main_output_dir = os.path.join(args.output_dir, f"output_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)

    input_path = args.input_dir if args.input_dir else args.input_file
    if not input_path or not os.path.exists(input_path):
        logging.error("No valid input path provided.")
        sys.exit(1)

    process_audio(
        input_path,
        main_output_dir,
        args.model,
        args.num_speakers,
        args.enable_vocal_separation
    )

if __name__ == "__main__":
    main()
