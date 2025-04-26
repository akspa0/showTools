import os
import sys
import argparse
from datetime import datetime
import logging
import json
import subprocess
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

VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv')

def normalize_audio(input_audio, output_dir, target_lufs=-16):
    """Normalize audio to target LUFS using ffmpeg."""
    normalized_dir = os.path.join(output_dir, "normalized")
    os.makedirs(normalized_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(input_audio))[0]
    output_file = os.path.join(normalized_dir, f"{base_name}_normalized.wav")
    
    # Use ffmpeg for consistent normalization
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
            
    # Return the dictionary containing detailed segment info, not just paths
    return segment_info

def transcribe_with_whisper(model, segment_info_dict, output_dir, enable_word_extraction=False):
    """Transcribe audio files using Whisper and create per-speaker transcripts."""
    all_transcriptions = {}
    word_timings = {}  # Initialize even if not used, for consistency
    word_counter = 0
    
    # Iterate through the segment info dictionary {speaker: [list_of_segment_dicts]}
    for speaker, segments in segment_info_dict.items():
        speaker_transcription_dir = os.path.join(output_dir, f"Speaker_{speaker}_transcriptions")
        os.makedirs(speaker_transcription_dir, exist_ok=True)
        
        # Create words directory for this speaker only if needed
        words_dir = None
        if enable_word_extraction:
            words_dir = os.path.join(output_dir, f"Speaker_{speaker}_words")
            os.makedirs(words_dir, exist_ok=True)
        
        speaker_full_transcript = ""
        segment_transcriptions = []
        
        # Sort segments by sequence number (already dictionaries)
        segments.sort(key=lambda x: x['sequence'])
        
        # Iterate through the list of segment dictionaries for this speaker
        for segment_info in segments:
            segment_path = segment_info['path']
            start_time = segment_info['start']
            end_time = segment_info['end']
            duration = segment_info['duration']
            seq_num = segment_info['sequence']
            segment_name = os.path.basename(segment_path) # Keep for logging

            try:
                # Skip very short segments that likely won't transcribe well
                if duration < 1:
                    continue
                
                # Transcribe the segment
                # Only request word timestamps if extraction is enabled
                transcription = model.transcribe(segment_path, word_timestamps=enable_word_extraction)
                text = transcription['text'].strip()
                
                if not text:
                    continue
                
                # Extract word timings only if enabled
                if enable_word_extraction and 'segments' in transcription and transcription['segments']:
                    for segment in transcription['segments']:
                        if 'words' in segment:
                            try: # Add try-except for audio loading robustness
                                audio = AudioSegment.from_file(segment_path)
                            except Exception as audio_load_err:
                                logging.warning(f"Could not load audio file {segment_path} for word extraction: {audio_load_err}")
                                continue # Skip word extraction for this segment if audio fails

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
                                    
                                # Extract word audio with generous padding
                                # Add substantial padding before and after the word (100ms or 40% of duration, whichever is greater)
                                padding_ms = max(100, int(word_duration * 1000 * 0.4))
                                
                                # Calculate padded boundaries (ensuring we don't go out of bounds)
                                word_start_ms = max(0, int((word_start * 1000) - padding_ms))
                                word_end_ms = min(len(audio), int((word_end * 1000) + padding_ms))
                                
                                # Extract with padding
                                word_audio = audio[word_start_ms:word_end_ms]
                                
                                # Apply very gentle fade in/out (10% of duration)
                                fade_ms = max(30, int(word_audio.duration_seconds * 1000 * 0.1))
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
                # Use exact start/end times from segment_info
                with open(transcription_file, "w", encoding='utf-8') as f: 
                    timestamp_str = f"[{start_time:.2f}s - {end_time:.2f}s]"
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
    
    # Save word timings to JSON (only if enabled and populated)
    if enable_word_extraction and word_timings:
        with open(os.path.join(output_dir, "word_timings.json"), 'w') as f:
            json.dump(word_timings, f, indent=2)
    elif enable_word_extraction:
        logging.info("Word extraction enabled, but no words were extracted.")
    
    # Create a master transcript with all speakers in chronological order
    all_segments = []
    for speaker, segments in all_transcriptions.items():
        all_segments.extend(segments)
    
    # Sort by start time (more reliable than sequence if passes differ)
    all_segments.sort(key=lambda x: x['start'])
    
    # Write master transcript
    master_transcript = ""
    for i, segment in enumerate(all_segments):
        timestamp = f"[{segment['start']:.2f}s - {segment['end']:.2f}s]"
        master_transcript += f"{i:04d} - SPEAKER {segment['speaker']}: {timestamp} {segment['text']}\n\n"
    
    if master_transcript:
        with open(os.path.join(output_dir, "master_transcript.txt"), 'w') as f:
            f.write(master_transcript)

def run_second_pass_diarization(first_pass_segment_info, first_pass_output_dir, diarization_pipeline, whisper_model, final_output_dir, segment_min_duration=5.0, second_pass_speakers=2):
    """
    Performs a second pass of diarization and transcription on segments 
    from the first pass to refine speaker separation.
    """
    logging.info("Starting Second Pass Diarization Refinement...")
    second_pass_base_dir = os.path.join(final_output_dir, "2nd_pass")
    second_pass_speakers_dir = os.path.join(second_pass_base_dir, "speakers")
    second_pass_transcripts_dir = os.path.join(second_pass_base_dir, "transcriptions")
    
    os.makedirs(second_pass_speakers_dir, exist_ok=True)
    os.makedirs(second_pass_transcripts_dir, exist_ok=True)
    
    refined_segments_info = [] # List to store info for the master transcript
    sub_segment_counter = 0 # Global counter for unique sub-segment filenames

    # Iterate through speakers and their first-pass segments (already dicts)
    for speaker, segments in first_pass_segment_info.items():
        logging.info(f"Processing Speaker {speaker} segments for second pass...")
        
        # Ensure speaker-specific directories exist in 2nd pass output
        current_spk_2nd_pass_audio_dir = os.path.join(second_pass_speakers_dir, f"Speaker_{speaker}")
        current_spk_2nd_pass_ts_dir = os.path.join(second_pass_transcripts_dir, f"Speaker_{speaker}_transcriptions")
        os.makedirs(current_spk_2nd_pass_audio_dir, exist_ok=True)
        os.makedirs(current_spk_2nd_pass_ts_dir, exist_ok=True)

        for segment_info in segments: # segments is now expected to be the list of dicts from slice_audio_by_speaker
            segment_path = segment_info['path']
            original_start_time = segment_info['start']
            original_end_time = segment_info['end']
            original_duration = segment_info['duration']
            original_sequence = segment_info['sequence']

            # Skip segments shorter than the minimum duration for refinement
            if original_duration < segment_min_duration:
                logging.debug(f"Skipping short segment {os.path.basename(segment_path)} ({original_duration:.2f}s)")
                continue

            logging.info(f"Analyzing segment: {os.path.basename(segment_path)} (Duration: {original_duration:.2f}s)")

            try:
                # Run diarization on this specific segment
                # Use the default or passed-in number of speakers for the second pass
                segment_diarization = diarization_pipeline(segment_path, num_speakers=second_pass_speakers)
                
                # Check if more than one speaker was significantly detected
                speaker_turns = {}
                for turn, _, spk_label in segment_diarization.itertracks(yield_label=True):
                    if spk_label not in speaker_turns:
                        speaker_turns[spk_label] = 0
                    speaker_turns[spk_label] += (turn.end - turn.start)
                
                significant_speakers = [spk for spk, dur in speaker_turns.items() if dur > 0.5] # Consider speakers with > 0.5s total duration significant

                if len(significant_speakers) <= 1:
                    logging.info(f"  -> No significant speaker overlap detected. Skipping refinement.")
                    continue # No refinement needed for this segment

                logging.info(f"  -> Refinement needed: Detected {len(significant_speakers)} speakers. Re-slicing and transcribing...")

                # Load the original segment audio
                segment_audio = AudioSegment.from_file(segment_path)

                # Re-slice based on the *new* diarization for this segment
                for turn, _, sub_speaker in segment_diarization.itertracks(yield_label=True):
                    sub_start = turn.start
                    sub_end = turn.end
                    sub_duration = sub_end - sub_start

                    # Skip very short sub-segments resulting from re-slicing
                    if sub_duration < 0.5: 
                        continue

                    # Extract sub-segment audio
                    # Timestamps are relative to the start of the *current* segment
                    sub_segment_audio = segment_audio[sub_start * 1000 : sub_end * 1000]

                    # Apply fade
                    fade_duration = min(50, sub_segment_audio.duration_seconds * 1000 / 4) 
                    sub_segment_audio = sub_segment_audio.fade_in(int(fade_duration)).fade_out(int(fade_duration))

                    # --- Create unique filename for the sub-segment ---
                    # Use absolute start time for sorting
                    abs_start = original_start_time + sub_start
                    abs_end = original_start_time + sub_end
                    
                    # Export the temporary refined audio sub-segment to transcribe it
                    # We need a temporary path because the final name depends on the transcription
                    temp_sub_segment_path_wav = os.path.join(current_spk_2nd_pass_audio_dir, f"temp_subsegment_{sub_segment_counter}.wav")
                    sub_segment_audio.export(temp_sub_segment_path_wav, format="wav")

                    # Transcribe the sub-segment (no word timestamps needed for 2nd pass)
                    try:
                        transcription = whisper_model.transcribe(temp_sub_segment_path_wav, word_timestamps=False)
                        text = transcription['text'].strip()
                        
                        # Remove temporary file after transcription
                        try:
                            os.remove(temp_sub_segment_path_wav)
                        except OSError as e:
                            logging.warning(f"Could not remove temporary sub-segment file {temp_sub_segment_path_wav}: {e}")

                        if text:
                            # --- Generate final filename based on content --- 
                            first_words = text.split()[:5]
                            # Add sub_segment_counter for uniqueness if first words are identical
                            base_name = f"{original_sequence:04d}_{sub_segment_counter:04d}_{sanitize_filename('_'.join(first_words))}"
                            
                            final_sub_segment_path_wav = os.path.join(current_spk_2nd_pass_audio_dir, f"{base_name}.wav")
                            final_sub_segment_path_txt = os.path.join(current_spk_2nd_pass_ts_dir, f"{base_name}.txt")
                            
                            # Rename the originally exported audio file (or re-export if needed, but rename is efficient)
                            # Re-exporting might be safer if temp removal failed? Let's re-export.
                            sub_segment_audio.export(final_sub_segment_path_wav, format="wav")
                            # --- End filename generation --- 

                            # Write transcription file with the new name
                            with open(final_sub_segment_path_txt, "w", encoding='utf-8') as f:
                                timestamp_str = f"[{abs_start:.2f}s - {abs_end:.2f}s]" # Use more precision
                                f.write(f"{timestamp_str} {text}")

                            # Store info for the final 2nd pass master transcript (using new paths)
                            refined_segments_info.append({
                                'speaker': sub_speaker, # The speaker label FROM THE SECOND PASS
                                'start': abs_start,
                                'end': abs_end,
                                'text': text,
                                'audio_file': final_sub_segment_path_wav, # Use new path
                                'transcript_file': final_sub_segment_path_txt, # Use new path
                                'sequence': sub_segment_counter # Still useful for internal tracking if needed
                            })
                        else:
                             logging.warning(f"  -> Sub-segment from {segment_path} at {abs_start:.2f}s resulted in empty transcription.")

                    except Exception as sub_transcribe_err:
                        logging.error(f"  -> Error transcribing sub-segment generated from {segment_path} at {abs_start:.2f}s: {sub_transcribe_err}")
                        # Clean up temp file even if transcription failed
                        if os.path.exists(temp_sub_segment_path_wav):
                            try:
                                os.remove(temp_sub_segment_path_wav)
                            except OSError as e:
                                logging.warning(f"Could not remove temporary sub-segment file {temp_sub_segment_path_wav} after error: {e}")

                    sub_segment_counter += 1

            except Exception as segment_process_err:
                logging.error(f"Error processing segment {segment_path} for second pass: {segment_process_err}")
                import traceback
                logging.error(traceback.format_exc())
                continue # Move to the next segment

    # Create the 2nd pass master transcript
    refined_segments_info.sort(key=lambda x: x['start']) # Sort by absolute start time

    master_transcript_2nd_pass = ""
    for i, segment in enumerate(refined_segments_info):
        timestamp = f"[{segment['start']:.2f}s - {segment['end']:.2f}s]"
        # Note: Speaker label here is from the second pass analysis
        master_transcript_2nd_pass += f"{i:04d} - SPEAKER {segment['speaker']} (Refined): {timestamp} {segment['text']}\n\n" 

    if master_transcript_2nd_pass:
        master_transcript_path = os.path.join(second_pass_base_dir, "master_transcript.txt")
        with open(master_transcript_path, 'w', encoding='utf-8') as f:
            f.write(master_transcript_2nd_pass)
        logging.info(f"Second pass master transcript saved to: {master_transcript_path}")
    else:
        logging.info("No refined segments generated during second pass.")

    logging.info("Second Pass Diarization Refinement Finished.")

def extract_audio_from_video(video_path, output_wav_path):
    """Extracts audio from video file using ffmpeg."""
    logging.info(f"Extracting audio from video: {video_path}")
    try:
        # Use ffmpeg to extract audio as 16-bit PCM WAV, 44.1kHz, mono
        # -vn: disable video recording
        # -acodec pcm_s16le: standard WAV audio codec
        # -ar 44100: audio sample rate
        # -ac 1: mono audio channel
        # -y: overwrite output file if exists
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
            output_wav_path
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True) # Added text=True
        logging.info(f"Successfully extracted audio to: {output_wav_path}")
        # logging.debug(f"ffmpeg output: {result.stdout}") # Optional: log ffmpeg output
        # logging.debug(f"ffmpeg stderr: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error extracting audio using ffmpeg: {e}")
        logging.error(f"ffmpeg command: {' '.join(e.cmd)}")
        logging.error(f"ffmpeg stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        logging.error("ffmpeg command not found. Please ensure ffmpeg is installed and in your PATH.")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during audio extraction: {e}")
        return False

def process_audio(input_path, output_dir, model_name, enable_vocal_separation, num_speakers, auto_speakers=True, enable_word_extraction=False, enable_second_pass=False):
    """Unified pipeline for audio processing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load models FIRST to avoid repeated loading
    logging.info(f"Loading Whisper model: {model_name}")
    model = load_model(model_name)
    
    logging.info("Loading diarization pipeline")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    pipeline.to(device)
    
    # Determine initial input file and base name
    initial_input_file = None
    input_basename = None
    extracted_audio_temp_file = None # Keep track if we extract audio

    if os.path.isfile(input_path):
        initial_input_file = input_path
        input_basename = os.path.splitext(os.path.basename(initial_input_file))[0]
    elif os.path.isdir(input_path):
        # If directory, find the most recent compatible file (audio or video)
        logging.info(f"Input is a directory: {input_path}. Searching for newest compatible file.")
        compatible_extensions = ('.mp3', '.wav', '.m4a', '.ogg', '.flac') + VIDEO_EXTENSIONS
        all_files = [
            f for f in os.listdir(input_path) 
            if os.path.isfile(os.path.join(input_path, f)) and f.lower().endswith(compatible_extensions)
        ]
        if not all_files:
            logging.error(f"No compatible audio or video files found in directory: {input_path}")
            return None
            
        all_files.sort(key=lambda x: os.path.getmtime(os.path.join(input_path, x)), reverse=True)
        newest_file = all_files[0]
        logging.info(f"Processing newest file found in directory: {newest_file}")
        initial_input_file = os.path.join(input_path, newest_file)
        input_basename = os.path.splitext(newest_file)[0]
    else:
        logging.error(f"Input path exists but is neither a file nor a directory: {input_path}")
        return None

    # Create output directory based on the original input base name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_output_dir = os.path.join(output_dir, f"{input_basename}_{timestamp}") 
    os.makedirs(final_output_dir, exist_ok=True)
    logging.info(f"Output directory set to: {final_output_dir}")

    audio_to_process = None
    try:
        # --- Check if input is video and extract audio --- 
        if initial_input_file.lower().endswith(VIDEO_EXTENSIONS):
            logging.info(f"Detected video input: {initial_input_file}")
            # Define path for temporary extracted audio within the output dir
            extracted_audio_temp_file = os.path.join(final_output_dir, f"{input_basename}_extracted_audio.wav")
            
            if not extract_audio_from_video(initial_input_file, extracted_audio_temp_file):
                logging.error("Audio extraction failed. Aborting processing.")
                return None # Stop if extraction fails
            
            audio_to_process = extracted_audio_temp_file # Use extracted audio for subsequent steps
        else:
            # It's an audio file, use it directly
            audio_to_process = initial_input_file
        
        logging.info(f"Audio file for processing: {audio_to_process}")
        # --- End Audio Extraction Check ---

        # 1. Normalize the audio (use the file determined above)
        normalized_file = normalize_audio(audio_to_process, final_output_dir)
        logging.info(f"Normalized audio: {normalized_file}")
        
        # The rest of the pipeline uses the *normalized* file path
        pipeline_audio_input = normalized_file

        # 2. Optional vocal separation
        if enable_vocal_separation:
            logging.info("Attempting vocal separation")
            try:
                # Check if demucs is installed
                try:
                    # Use subprocess.run consistently
                    result = subprocess.run(["demucs", "--version"], capture_output=True, check=True, text=True)
                    logging.debug(f"Demucs version check output: {result.stdout}")
                    has_demucs = True
                except (subprocess.SubprocessError, FileNotFoundError):
                    has_demucs = False
                    logging.warning("Demucs command not found or failed. Skipping vocal separation.")
                
                if has_demucs:
                    vocals_file = separate_vocals_with_demucs(pipeline_audio_input, final_output_dir) # Use normalized file
                    if vocals_file and os.path.exists(vocals_file):
                        pipeline_audio_input = vocals_file # Update input for next steps
                        logging.info(f"Using separated vocals: {vocals_file}")
                    else:
                        logging.warning("Vocal separation failed or produced no output, using normalized audio for subsequent steps.")
            except Exception as e:
                logging.warning(f"Vocal separation error: {e}. Using normalized audio for subsequent steps.")
        
        # 3. Speaker diarization (use the potentially separated vocals)
        speaker_output_dir = os.path.join(final_output_dir, "speakers")
        os.makedirs(speaker_output_dir, exist_ok=True)
        
        actual_num_speakers = num_speakers
        if auto_speakers:
            try:
                actual_num_speakers = detect_optimal_speakers(pipeline, pipeline_audio_input)
            except Exception as e:
                logging.warning(f"Auto speaker detection failed: {e}. Using provided value: {num_speakers}")
                actual_num_speakers = num_speakers
        
        logging.info(f"Running diarization with {actual_num_speakers} speakers on {pipeline_audio_input}")
        diarization = pipeline(pipeline_audio_input, num_speakers=actual_num_speakers)
        
        # 4. Slice audio by speaker (use the potentially separated vocals)
        segment_info_dict = slice_audio_by_speaker(pipeline_audio_input, diarization, speaker_output_dir)
        logging.info(f"Sliced audio into speakers: {list(segment_info_dict.keys())}")
        
        # 5. Transcribe speaker segments
        logging.info("Starting transcription")
        transcribe_with_whisper(model, segment_info_dict, final_output_dir, enable_word_extraction=enable_word_extraction)
        logging.info("Transcription complete")

        # 5b. Optional Second Pass Refinement
        if enable_second_pass:
            logging.info("Starting second pass refinement...")
            run_second_pass_diarization(
                first_pass_segment_info=segment_info_dict,
                first_pass_output_dir=speaker_output_dir, 
                diarization_pipeline=pipeline,
                whisper_model=model,
                final_output_dir=final_output_dir 
            )
            logging.info("Second pass refinement complete.")
        else:
            logging.info("Second pass refinement skipped.")
        
        # 6. Zip results
        # Use the original input basename for the zip file name
        zip_file = zip_results(final_output_dir, initial_input_file) 
        logging.info(f"Results zipped to: {zip_file}")
        
        return final_output_dir
        
    except Exception as e:
        # Catch general processing errors after audio extraction
        logging.error(f"Error processing {audio_to_process}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None
    finally:
        # Cleanup: Remove temporary extracted audio file if it was created
        if extracted_audio_temp_file and os.path.exists(extracted_audio_temp_file):
            try:
                logging.info(f"Cleaning up temporary extracted audio file: {extracted_audio_temp_file}")
                os.remove(extracted_audio_temp_file)
            except OSError as e:
                logging.warning(f"Could not remove temporary audio file {extracted_audio_temp_file}: {e}")

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
    parser.add_argument('--enable_word_extraction', action='store_true', help='Enable extraction of individual word audio snippets.')
    parser.add_argument('--enable_second_pass', action='store_true', help='Enable second pass diarization for refinement (experimental).')
    args = parser.parse_args()

    # Restore original input validation
    if not any([args.input_dir, args.input_file, args.url]):
        logging.error("Please provide one of --input_dir, --input_file, or --url.")
        parser.print_help()
        sys.exit(1)
    # Ensure only one input type is provided (optional, but good practice)
    input_args = [args.input_dir, args.input_file, args.url]
    if sum(1 for arg in input_args if arg is not None) > 1:
         logging.error("Please provide only one type of input: --input_dir, --input_file, or --url.")
         parser.print_help()
         sys.exit(1)

    # Restore original input path determination logic
    # Specific output dir determination remains inside process_audio
    input_path_for_processing = None
    if args.url:
        # For CLI, download first to a temp/output location, then pass the file path.
        download_target_dir = os.path.join(args.output_dir, "downloads") # Download to a subdir
        os.makedirs(download_target_dir, exist_ok=True)
        try:
            logging.info(f"Downloading audio from {args.url} to {download_target_dir}")
            input_path_for_processing = download_audio(args.url, download_target_dir)
            if not input_path_for_processing:
                 raise ValueError("download_audio failed to return a valid path.")
        except Exception as e:
            logging.error(f"Failed to download URL {args.url}: {e}")
            sys.exit(1)
    elif args.input_dir:
        input_path_for_processing = args.input_dir # Pass the directory path
    else: # Must be input_file
        input_path_for_processing = args.input_file

    # Final check on the path before processing
    if not input_path_for_processing or not os.path.exists(input_path_for_processing):
        logging.error(f"Input path does not exist or could not be determined: {input_path_for_processing}")
        sys.exit(1)
    # No isfile check here, allow directories to be passed

    # Call process_audio with the determined path and the main output dir
    process_audio(
        input_path_for_processing, 
        args.output_dir, # Pass the user-specified main output directory
        args.model,
        args.enable_vocal_separation,
        args.num_speakers,
        args.auto_speakers,
        args.enable_word_extraction,
        args.enable_second_pass
    )

if __name__ == "__main__":
    main()
