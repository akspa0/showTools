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
    segment_counter = 0
    
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
            
            # Format: Speaker_X/0000_segment_START_END_DURATION.wav
            segment_filename = f"{segment_counter:04d}_segment_{int(segment['start'])}_{int(segment['end'])}_{int(segment['duration'])}.wav"
            segment_path = os.path.join(speaker_dir, segment_filename)
            
            # Export with normalized volume
            segment_audio.export(segment_path, format="wav")
            
            speaker_files[speaker].append(segment_path)
            segment_info[speaker].append({
                'path': segment_path,
                'start': segment['start'],
                'end': segment['end'],
                'duration': segment['duration'],
                'sequence': segment_counter
            })
            
            segment_counter += 1
            
    # Create a JSON file with all segment information (useful for debugging/analytics)
    with open(os.path.join(speaker_output_dir, "segments.json"), 'w') as f:
        json.dump(segment_info, f, indent=2)
            
    return speaker_files

def transcribe_with_whisper(model, speaker_files, output_dir):
    """Transcribe audio files using Whisper and create per-speaker transcripts."""
    all_transcriptions = {}
    word_timings = {}
    word_counter = 0
    
    for speaker, segments in speaker_files.items():
        speaker_transcription_dir = os.path.join(output_dir, f"Speaker_{speaker}_transcriptions")
        os.makedirs(speaker_transcription_dir, exist_ok=True)
        
        # Create words directory for this speaker
        words_dir = os.path.join(output_dir, f"Speaker_{speaker}_words")
        os.makedirs(words_dir, exist_ok=True)
        
        speaker_full_transcript = ""
        segment_transcriptions = []
        
        # Sort segments by sequence number in filename
        segments.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))
        
        for segment_path in segments:
            try:
                # Get info from filename
                segment_name = os.path.basename(segment_path)
                parts = segment_name.split('_')
                
                # Parse sequence number and timing info
                try:
                    seq_num = int(parts[0])
                    
                    # Extract timing information - handle different filename formats
                    if len(parts) >= 5 and parts[1] == "segment":
                        # Format: 0000_segment_START_END_DURATION.wav
                        start_time = int(parts[2])
                        end_time = int(parts[3])
                        # Get duration without the .wav extension
                        duration_str = parts[4].split('.')[0]
                        duration = int(duration_str)
                    else:
                        # Fallback if filename format is unexpected
                        logging.warning(f"Unexpected filename format: {segment_name}, using defaults")
                        start_time = 0
                        end_time = 10
                        duration = 10
                except (ValueError, IndexError) as e:
                    logging.warning(f"Error parsing filename {segment_name}: {e}, using defaults")
                    seq_num = 0
                    start_time = 0
                    end_time = 10
                    duration = 10
                start_time = int(segment_name.split('_')[2])
                
                # Skip very short segments that likely won't transcribe well
                if duration < 1:
                    continue
                
                # Transcribe the segment with word timestamps
                transcription = model.transcribe(segment_path, word_timestamps=True)
                text = transcription['text'].strip()
                
                if not text:
                    continue
                
                # Extract word timings
                if 'segments' in transcription and transcription['segments']:
                    for segment in transcription['segments']:
                        if 'words' in segment:
                            audio = AudioSegment.from_file(segment_path)
                            
                            for word_data in segment['words']:
                                word = word_data['word'].strip()
                                if not word:
                                    continue
                                    
                                # Get word timing
                                word_start = word_data['start']
                                word_end = word_data['end']
                                word_duration = word_end - word_start
                                
                                # Skip very short words
                                if word_duration < 0.1:
                                    continue
                                    
                                # Extract word audio
                                word_audio = audio[word_start * 1000:word_end * 1000]
                                
                                # Apply fade in/out
                                fade_ms = min(30, int(word_duration * 1000 / 3))
                                word_audio = word_audio.fade_in(fade_ms).fade_out(fade_ms)
                                
                                # Save word audio
                                word_filename = f"{word_counter:04d}_{word.replace(' ', '_')}.wav"
                                word_path = os.path.join(words_dir, word_filename)
                                word_audio.export(word_path, format="wav")
                                
                                # Store word timing info
                                if speaker not in word_timings:
                                    word_timings[speaker] = []
                                    
                                word_timings[speaker].append({
                                    'word': word,
                                    'file': word_path,
                                    'start': start_time + word_start,
                                    'end': start_time + word_end,
                                    'duration': word_duration,
                                    'sequence': word_counter
                                })
                                
                                word_counter += 1
                
                # Create a reasonable output filename based on the first few words and sequence
                first_words = text.split()[:5]
                base_name = f"{seq_num:04d}_{sanitize_filename('_'.join(first_words))}"
                
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
                    'transcript_file': transcription_file,
                    'sequence': seq_num
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
    
    # Save word timings to JSON
    with open(os.path.join(output_dir, "word_timings.json"), 'w') as f:
        json.dump(word_timings, f, indent=2)
    
    # Create a master transcript with all speakers in chronological order
    all_segments = []
    for speaker, segments in all_transcriptions.items():
        all_segments.extend(segments)
    
    # Sort by sequence number
    all_segments.sort(key=lambda x: x['sequence'])
    
    # Write master transcript
    master_transcript = ""
    for i, segment in enumerate(all_segments):
        timestamp = f"[{segment['start']}s - {segment['end']}s]"
        master_transcript += f"{i:04d} - SPEAKER {segment['speaker']}: {timestamp} {segment['text']}\n\n"
    
    if master_transcript:
        with open(os.path.join(output_dir, "master_transcript.txt"), 'w') as f:
            f.write(master_transcript)

def process_audio(input_path, output_dir, model_name, enable_vocal_separation, num_speakers, auto_speakers=True):
    """Unified pipeline for audio processing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load models FIRST to avoid repeated loading
    logging.info(f"Loading Whisper model: {model_name}")
    model = load_model(model_name)
    
    logging.info("Loading diarization pipeline")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    pipeline.to(device)
    
    # Determine which audio file to process
    if os.path.isfile(input_path):
        input_basename = os.path.splitext(os.path.basename(input_path))[0]
        audio_to_process = input_path
    else:
        # If URL was provided, the downloaded file is the input
        logging.error("Expected a file path, got directory. Using most recent file.")
        all_files = [
            f for f in os.listdir(input_path) 
            if os.path.isfile(os.path.join(input_path, f)) and f.endswith(('.mp3', '.wav', '.m4a'))
        ]
        if not all_files:
            logging.error("No audio files found.")
            return
            
        # Use most recently modified file
        all_files.sort(key=lambda x: os.path.getmtime(os.path.join(input_path, x)), reverse=True)
        newest_file = all_files[0]
        input_basename = os.path.splitext(newest_file)[0]
        audio_to_process = os.path.join(input_path, newest_file)
    
    # Create output directory for this specific audio file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_output_dir = os.path.join(output_dir, f"{input_basename}_{timestamp}")
    os.makedirs(final_output_dir, exist_ok=True)
    logging.info(f"Processing audio: {audio_to_process}")
    logging.info(f"Output directory: {final_output_dir}")
    
    try:
        # 1. Normalize the audio
        normalized_file = normalize_audio(audio_to_process, final_output_dir)
        logging.info(f"Normalized audio: {normalized_file}")
        
        # 2. Optional vocal separation
        if enable_vocal_separation:
            logging.info("Attempting vocal separation")
            try:
                # Check if demucs is installed
                try:
                    subprocess.run(["demucs", "--version"], 
                                  capture_output=True, check=True)
                    has_demucs = True
                except (subprocess.SubprocessError, FileNotFoundError):
                    has_demucs = False
                    logging.warning("Demucs not found. Skipping vocal separation.")
                
                if has_demucs:
                    vocals_file = separate_vocals_with_demucs(normalized_file, final_output_dir)
                    if os.path.exists(vocals_file):
                        audio_to_process = vocals_file
                        logging.info(f"Using separated vocals: {vocals_file}")
                    else:
                        logging.warning("Vocal separation failed, using normalized audio")
                        audio_to_process = normalized_file
                else:
                    audio_to_process = normalized_file
            except Exception as e:
                logging.warning(f"Vocal separation error: {e}. Using normalized audio.")
                audio_to_process = normalized_file
        else:
            audio_to_process = normalized_file
        
        # 3. Speaker diarization
        speaker_output_dir = os.path.join(final_output_dir, "speakers")
        os.makedirs(speaker_output_dir, exist_ok=True)
        
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
        logging.info(f"Sliced audio into speakers: {list(speaker_files.keys())}")
        
        # 5. Transcribe speaker segments
        logging.info("Starting transcription")
        transcribe_with_whisper(model, speaker_files, final_output_dir)
        logging.info("Transcription complete")
        
        # 6. Zip results
        zip_file = zip_results(final_output_dir, audio_to_process)
        logging.info(f"Results zipped to: {zip_file}")
        
        return final_output_dir
        
    except Exception as e:
        logging.error(f"Error processing {audio_to_process}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def copy_final_outputs(temp_dir, final_output_dir):
    """Copy only the final output files to the final output directory."""
    # Copy only necessary files, not intermediate ones
    files_to_copy = [
        "master_transcript.txt",
        "word_timings.json"
    ]
    
    directories_to_copy = [
        "*_transcriptions",
        "*_words",
    ]
    
    for file in files_to_copy:
        src = os.path.join(temp_dir, file)
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, os.path.join(final_output_dir, file))
    
    import glob
    for pattern in directories_to_copy:
        for dir_path in glob.glob(os.path.join(temp_dir, pattern)):
            dir_name = os.path.basename(dir_path)
            target_dir = os.path.join(final_output_dir, dir_name)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            # Copy all files in the directory
            for file in os.listdir(dir_path):
                src = os.path.join(dir_path, file)
                if os.path.isfile(src):
                    import shutil
                    shutil.copy2(src, os.path.join(target_dir, file))
    
    # Copy full transcripts
    for file in os.listdir(temp_dir):
        if file.endswith("_full_transcript.txt"):
            src = os.path.join(temp_dir, file)
            import shutil
            shutil.copy2(src, os.path.join(final_output_dir, file))

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
