import os
import sys
import argparse
from datetime import datetime
import logging
from whisper import load_model
from pyannote.audio import Pipeline
import torch
from utils import (
    sanitize_filename, download_audio, zip_results, convert_and_normalize_audio,
    slice_on_voice_activity, slice_by_word_timestamps, normalize_audio_ffmpeg
)
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

def transcribe_with_whisper(model, audio_files, output_dir, word_timestamps=False):
    """Transcribe audio files using Whisper with optional word-level timestamps."""
    results = []
    for audio_file in audio_files:
        if not os.path.exists(audio_file):
            logging.error(f"Audio file not found: {audio_file}")
            continue

        try:
            logging.info(f"Transcribing file: {audio_file}")
            # Configure Whisper options for word timestamps
            transcription = model.transcribe(
                audio_file,
                word_timestamps=True if word_timestamps else None,
                language=None,  # Auto-detect language
                verbose=True if word_timestamps else None  # Show progress for word timestamps
            )
            
            # Log the structure of the transcription for debugging
            if word_timestamps:
                logging.debug("Transcription structure:")
                for key in transcription.keys():
                    logging.debug(f"Key: {key}")
                    if key == 'segments':
                        for i, segment in enumerate(transcription['segments']):
                            logging.debug(f"Segment {i} keys: {segment.keys()}")
                            if 'words' in segment:
                                logging.debug(f"Words data available in segment {i}")
            
            transcription_text = transcription.get('text', '').strip()

            # Create a base name for the files
            base_name = sanitize_filename(transcription_text[:50] if transcription_text else "transcription")
            transcription_file = os.path.join(output_dir, f"{base_name}.txt")
            timestamps_file = os.path.join(output_dir, f"{base_name}_timestamps.json")

            # Save the full transcription text
            with open(transcription_file, "w", encoding='utf-8') as f:
                f.write(transcription_text)

            # Save the detailed segments with timestamps
            import json
            with open(timestamps_file, "w", encoding='utf-8') as f:
                json.dump(transcription.get('segments', []), f, indent=2, ensure_ascii=False)

            logging.info(f"Transcription saved to {transcription_file}")
            logging.info(f"Timestamps saved to {timestamps_file}")
            
            results.append(transcription)
        except Exception as e:
            logging.error(f"Error transcribing {audio_file}: {e}")
            continue
    
    # Return the last transcription if any were successful
    return results[-1] if results else None

def process_audio(input_path, output_dir, model_name, enable_vocal_separation, num_speakers, use_voice_activation=False, enable_word_timestamps=False):
    """Unified pipeline for audio processing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
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
                    processed_file = vocals_file
                else:
                    logging.error(f"Vocal-separated file {vocals_file} not found.")
                    continue
            else:
                processed_file = normalized_file

            model = load_model(model_name)
            
            if use_voice_activation:
                # Use voice activity detection for slicing
                logging.info("Using voice activation-based slicing")
                segments = slice_on_voice_activity(processed_file, base_output_dir)
                
                # Process each voice segment
                for segment in segments:
                    segment_dir = os.path.join(base_output_dir, "voice_segments_transcriptions")
                    os.makedirs(segment_dir, exist_ok=True)
                    
                    # Normalize the segment and keep track of both original and normalized paths
                    normalized_segment = normalize_audio_ffmpeg(segment)
                    
                    # Transcribe with word timestamps if enabled
                    transcription = transcribe_with_whisper(model, [normalized_segment], segment_dir, enable_word_timestamps)
                    
                    if enable_word_timestamps and transcription:
                        try:
                            # Use the normalized segment for word-level slicing
                            logging.info(f"Processing word-level slicing for {normalized_segment}")
                            slice_by_word_timestamps(normalized_segment, transcription['segments'], segment_dir)
                        except Exception as e:
                            logging.error(f"Error during word-level slicing of {normalized_segment}: {e}")
            else:
                # Use speaker diarization
                logging.info(f"Using speaker diarization with {num_speakers} expected speakers")
                pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
                pipeline.to(device)
                
                speaker_output_dir = os.path.join(base_output_dir, "speakers")
                diarization = pipeline(processed_file)
                speaker_files = slice_audio_by_speaker(processed_file, diarization, speaker_output_dir)

                for speaker, segments in speaker_files.items():
                    speaker_transcription_dir = os.path.join(base_output_dir, f"Speaker_{speaker}_transcriptions")
                    os.makedirs(speaker_transcription_dir, exist_ok=True)

                    # Normalize and boost soundbites after slicing
                    normalized_segments = []
                    for segment in segments:
                        normalized_segment = normalize_audio_ffmpeg(segment)
                        normalized_segments.append(normalized_segment)

                    # Transcribe with word timestamps if enabled
                    transcription = transcribe_with_whisper(model, normalized_segments, speaker_transcription_dir, enable_word_timestamps)
                    
                    # Process each segment for word-level slicing
                    if enable_word_timestamps and transcription and 'segments' in transcription:
                        for segment_file, segment_transcription in zip(normalized_segments, transcription['segments']):
                            try:
                                if os.path.exists(segment_file):
                                    logging.info(f"Processing word-level slicing for {segment_file}")
                                    slice_by_word_timestamps(segment_file, [segment_transcription], speaker_transcription_dir)
                                else:
                                    logging.error(f"Audio file not found for word-level slicing: {segment_file}")
                            except Exception as e:
                                logging.error(f"Error during word-level slicing of {segment_file}: {e}")

        except Exception as e:
            logging.error(f"Error processing {audio_file}: {e}")
            continue

        # Zip the results
        zip_results(base_output_dir, audio_file)

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
    parser.add_argument('--use_voice_activation', action='store_true', 
                       help='Use voice activation-based slicing instead of speaker diarization.')
    parser.add_argument('--enable_word_timestamps', action='store_true',
                       help='Enable word-level timestamps and slicing.')
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
        args.num_speakers,
        args.use_voice_activation,
        args.enable_word_timestamps
    )

if __name__ == "__main__":
    main()
