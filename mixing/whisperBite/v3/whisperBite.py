import os
import sys
import argparse
from datetime import datetime
import logging
import json
from whisper import load_model
from pyannote.audio import Pipeline
import torch
from pydub import AudioSegment
from utils import sanitize_filename, download_audio, zip_results
from vocal_separation import separate_vocals_with_demucs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def normalize_audio(input_audio, output_dir, target_lufs=-16):
    """Normalize audio to target LUFS using ffmpeg."""
    normalized_dir = os.path.join(output_dir, "normalized")
    os.makedirs(normalized_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(input_audio))[0]
    output_file = os.path.join(normalized_dir, f"{base_name}_normalized.wav")
    
    # Use ffmpeg for consistent normalization
    import subprocess
    cmd = [
        "ffmpeg", "-y", "-i", input_audio,
        "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
        "-ar", "44100", "-acodec", "pcm_s16le", output_file
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        logging.info(f"Normalized audio saved to {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        logging.error(f"Error normalizing audio: {e}")
        logging.error(f"STDERR: {e.stderr.decode('utf-8')}")
        raise

def detect_optimal_speakers(diarization_pipeline, audio_file, min_speakers=1, max_speakers=10):
    """Find optimal speaker count using clustering quality metrics."""
    best_score = -float('inf')
    best_num_speakers = 2  # Default fallback
    
    # Try different speaker counts and use silhouette score to evaluate
    for num_speakers in range(min_speakers, min(max_speakers + 1, 6)):  # Limit to reasonable range
        try:
            diarization = diarization_pipeline(audio_file, num_speakers=num_speakers)
            
            # Extract speaker segments for clustering quality assessment
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'duration': turn.end - turn.start,
                    'speaker': speaker
                })
            
            # Simplistic quality metric: longer segments are better (less fragmentation)
            avg_segment_duration = sum(s['duration'] for s in segments) / len(segments) if segments else 0
            num_speaker_changes = len(segments) - 1
            
            # Penalize excessive fragmentation while promoting longer segments
            score = avg_segment_duration * 10 - num_speaker_changes * 0.1
            
            logging.info(f"Speaker count {num_speakers}: score={score:.2f} (avg_duration={avg_segment_duration:.2f}s, changes={num_speaker_changes})")
            
            if score > best_score:
                best_score = score
                best_num_speakers = num_speakers
        
        except Exception as e:
            logging.warning(f"Error evaluating {num_speakers} speakers: {e}")
            continue
    
    logging.info(f"Selected optimal speaker count: {best_num_speakers}")
    return best_num_speakers

def slice_audio_by_speaker(file_path, diarization, speaker_output_dir, min_segment_duration=1.0):
    """Slice audio by speakers based on diarization results."""
    audio = AudioSegment.from_file(file_path)
    os.makedirs(speaker_output_dir, exist_ok=True)

    # Group segments by speaker
    speaker_segments = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        duration = turn.end - turn.start
        
        # Skip very short segments
        if duration < min_segment_duration:
            continue
            
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
            
        speaker_segments[speaker].append({
            'start': turn.start,
            'end': turn.end,
            'duration': duration
        })

    # Create speaker directories and slice audio
    speaker_files = {}
    segment_info = {}
    
    for speaker, segments in speaker_segments.items():
        speaker_dir = os.path.join(speaker_output_dir, f"Speaker_{speaker}")
        os.makedirs(speaker_dir, exist_ok=True)
        
        if speaker not in speaker_files:
            speaker_files[speaker] = []
            segment_info[speaker] = []
        
        # Sort segments by start time
        segments.sort(key=lambda x: x['start'])
        
        # Merge very close segments from the same speaker (fix over-segmentation)
        merged_segments = []
        if segments:
            current = segments[0]
            for next_segment in segments[1:]:
                gap = next_segment['start'] - current['end']
                
                # If gap is small (less than 0.5s), merge segments
                if gap < 0.5:
                    current['end'] = next_segment['end']
                    current['duration'] = current['end'] - current['start']
                else:
                    merged_segments.append(current)
                    current = next_segment
                    
            merged_segments.append(current)
        
        # Extract and save the merged segments
        for i, segment in enumerate(merged_segments):
            segment_audio = audio[segment['start'] * 1000:segment['end'] * 1000]
            
            # Apply a slight fade in/out to avoid clicks
            fade_duration = min(100, segment_audio.duration_seconds * 1000 / 4)  # 100ms or 1/4 of segment
            segment_audio = segment_audio.fade_in(int(fade_duration)).fade_out(int(fade_duration))
            
            # Format: Speaker_X/segment_START_END_DURATION.wav
            segment_filename = f"segment_{int(segment['start'])}_{int(segment['end'])}_{int(segment['duration'])}.wav"
            segment_path = os.path.join(speaker_dir, segment_filename)
            
            # Export with normalized volume
            segment_audio.export(segment_path, format="wav")
            
            speaker_files[speaker].append(segment_path)
            segment_info[speaker].append({
                'path': segment_path,
                'start': segment['start'],
                'end': segment['end'],
                'duration': segment['duration']
            })
            
    # Create a JSON file with all segment information (useful for debugging/analytics)
    with open(os.path.join(speaker_output_dir, "segments.json"), 'w') as f:
        json.dump(segment_info, f, indent=2)
            
    return speaker_files

def transcribe_with_whisper(model, speaker_files, output_dir):
    """Transcribe audio files using Whisper and create per-speaker transcripts."""
    all_transcriptions = {}
    
    for speaker, segments in speaker_files.items():
        speaker_transcription_dir = os.path.join(output_dir, f"Speaker_{speaker}_transcriptions")
        os.makedirs(speaker_transcription_dir, exist_ok=True)
        
        speaker_full_transcript = ""
        segment_transcriptions = []
        
        # Sort segments by filename (which contains start time)
        segments.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))
        
        for segment_path in segments:
            try:
                # Get timing info from filename
                segment_name = os.path.basename(segment_path)
                start_time, end_time, duration = map(int, segment_name.replace("segment_", "").replace(".wav", "").split("_"))
                
                # Skip very short segments that likely won't transcribe well
                if duration < 1:
                    continue
                
                # Transcribe the segment
                transcription = model.transcribe(segment_path)
                text = transcription['text'].strip()
                
                if not text:
                    continue
                    
                # Create a reasonable output filename based on the first few words
                first_words = text.split()[:5]
                base_name = sanitize_filename("_".join(first_words))
                
                # Create output paths
                transcription_file = os.path.join(speaker_transcription_dir, f"{base_name}.txt")
                audio_output_file = os.path.join(speaker_transcription_dir, f"{base_name}.wav")
                
                # Write transcription to file
                with open(transcription_file, "w") as f:
                    timestamp_str = f"[{start_time}s - {end_time}s]"
                    f.write(f"{timestamp_str} {text}")
                
                # Copy audio file with the new name
                import shutil
                shutil.copy2(segment_path, audio_output_file)
                
                # Add to the full transcript for this speaker
                speaker_full_transcript += f"{timestamp_str} {text}\n\n"
                
                # Store info for master transcript
                segment_transcriptions.append({
                    'speaker': speaker,
                    'start': start_time,
                    'end': end_time,
                    'text': text,
                    'audio_file': audio_output_file,
                    'transcript_file': transcription_file
                })
                
            except Exception as e:
                logging.error(f"Error transcribing segment {segment_path}: {e}")
                continue
        
        # Write the combined transcript for this speaker
        if speaker_full_transcript:
            with open(os.path.join(output_dir, f"Speaker_{speaker}_full_transcript.txt"), 'w') as f:
                f.write(f"=== SPEAKER {speaker} TRANSCRIPT ===\n\n")
                f.write(speaker_full_transcript)
        
        all_transcriptions[speaker] = segment_transcriptions
    
    # Create a master transcript with all speakers in chronological order
    all_segments = []
    for speaker, segments in all_transcriptions.items():
        all_segments.extend(segments)
    
    # Sort by start time
    all_segments.sort(key=lambda x: x['start'])
    
    # Write master transcript
    master_transcript = ""
    for segment in all_segments:
        timestamp = f"[{segment['start']}s - {segment['end']}s]"
        master_transcript += f"SPEAKER {segment['speaker']}: {timestamp} {segment['text']}\n\n"
    
    if master_transcript:
        with open(os.path.join(output_dir, "master_transcript.txt"), 'w') as f:
            f.write(master_transcript)

def process_audio(input_path, output_dir, model_name, enable_vocal_separation, num_speakers, auto_speakers=True):
    """Unified pipeline for audio processing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Clear any existing processed files in output directories
    # to avoid confusion with previous runs
    for subdir in os.listdir(output_dir):
        subdir_path = os.path.join(output_dir, subdir)
        if os.path.isdir(subdir_path) and subdir not in ["normalized", "demucs", "speakers"]:
            for file in os.listdir(subdir_path):
                if file.endswith("_results.zip"):
                    # Keep zip files as they're final outputs
                    continue
                file_path = os.path.join(subdir_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)

    # Determine which audio files to process
    if os.path.isfile(input_path):
        audio_files = [input_path]
    else:
        # Process only the most recently modified audio files if in a directory
        all_files = [
            os.path.join(input_path, f) for f in os.listdir(input_path) 
            if f.endswith(('.mp3', '.wav', '.m4a'))
        ]
        # Sort by modification time (newest first)
        all_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # If there are too many files, just process the newest one
        audio_files = all_files[:1] if all_files else []
        
        if not audio_files:
            logging.error("No audio files found in the specified directory.")
            return

    # Load models
    logging.info(f"Loading Whisper model: {model_name}")
    model = load_model(model_name)
    
    logging.info("Loading diarization pipeline")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    pipeline.to(device)

    for audio_file in audio_files:
        logging.info(f"Processing file: {audio_file}")
        base_output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(audio_file))[0])
        os.makedirs(base_output_dir, exist_ok=True)

        # Processing steps with better error handling
        try:
            # 1. Normalize the audio (single normalization step)
            normalized_file = normalize_audio(audio_file, base_output_dir)
            
            # 2. Optional vocal separation
            if enable_vocal_separation:
                logging.info("Performing vocal separation")
                try:
                    # Check if demucs is installed first
                    try:
                        subprocess.run(["demucs", "--version"], 
                                      capture_output=True, check=True)
                        has_demucs = True
                    except (subprocess.SubprocessError, FileNotFoundError):
                        has_demucs = False
                        logging.warning("Demucs not found. Skipping vocal separation.")
                    
                    if has_demucs:
                        vocals_file = separate_vocals_with_demucs(normalized_file, base_output_dir)
                        if os.path.exists(vocals_file):
                            audio_to_process = vocals_file
                        else:
                            logging.warning("Vocal separation failed, falling back to normalized audio")
                            audio_to_process = normalized_file
                    else:
                        audio_to_process = normalized_file
                except Exception as e:
                    logging.warning(f"Vocal separation error: {e}. Falling back to normalized audio.")
                    audio_to_process = normalized_file
            else:
                audio_to_process = normalized_file
            
            # 3. Speaker diarization (with optional automatic speaker detection)
            speaker_output_dir = os.path.join(base_output_dir, "speakers")
            
            actual_num_speakers = num_speakers
            if auto_speakers:
                try:
                    actual_num_speakers = detect_optimal_speakers(pipeline, audio_to_process)
                except Exception as e:
                    logging.warning(f"Auto speaker detection failed: {e}. Using provided value: {num_speakers}")
                    actual_num_speakers = num_speakers
            
            logging.info(f"Running diarization with {actual_num_speakers} speakers")
            diarization = pipeline(audio_to_process, num_speakers=actual_num_speakers)
            
            # 4. Slice audio by speaker
            speaker_files = slice_audio_by_speaker(audio_to_process, diarization, speaker_output_dir)
            
            # 5. Transcribe speaker segments
            transcribe_with_whisper(model, speaker_files, base_output_dir)
            
            # 6. Zip results
            zip_results(base_output_dir, audio_file)
            
            logging.info(f"Successfully processed {audio_file}")
            
        except Exception as e:
            logging.error(f"Error processing {audio_file}: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(
        description="WhisperBite: Audio processing for transcription and speaker diarization."
    )
    parser.add_argument('--input_dir', type=str, help='Directory containing input audio files.')
    parser.add_argument('--input_file', type=str, help='Single audio file for processing.')
    parser.add_argument('--url', type=str, help='URL to download audio from.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files.')
    parser.add_argument('--model', type=str, default="base", help='Whisper model to use (default: base).')
    parser.add_argument('--num_speakers', type=int, default=2, help='Number of speakers for diarization (default: 2).')
    parser.add_argument('--auto_speakers', action='store_true', help='Automatically detect optimal speaker count.')
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
        args.num_speakers,
        args.auto_speakers
    )

if __name__ == "__main__":
    main()
