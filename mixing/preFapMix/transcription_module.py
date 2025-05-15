import logging
from pathlib import Path
import json
import torch
from whisper import load_model as whisper_load_model # Renamed to avoid conflict
from pyannote.core import Annotation
from pyannote.database.util import RTTMParser # Added for RTTM parsing
from pydub import AudioSegment # For slicing
import tempfile
import shutil # For cleaning up temp slice dir

logger = logging.getLogger(__name__)

# Adapted from WhisperBite.utils - simplified for speaker labels like SPEAKER_00, S0 etc.
def format_speaker_label_for_output(label_from_pyannote: str) -> str:
    """Formats pyannote speaker labels (e.g., SPEAKER_00) to S0, S1 for consistent output."""
    try:
        # Assuming labels like SPEAKER_00, SPEAKER_01, etc. or integer-like from pyannote
        return f"S{int(label_from_pyannote.split('_')[-1])}"
    except ValueError: # If it's already S0, S1 or just an int string
        try:
            return f"S{int(label_from_pyannote)}"
        except ValueError:
            return label_from_pyannote # Fallback to original if not recognized pattern


# Adapted from WhisperBite.whisperBite.slice_audio_by_speaker
def slice_audio_for_transcription(
    audio_file_path: str, 
    diarization_annotation: Annotation, 
    temp_slices_dir: Path,
    pii_safe_file_prefix: str, # Added for PII-safe slice naming
    min_segment_duration_ms: int = 500 # Min duration in ms to avoid tiny slices
):
    """
    Slices an audio file based on pyannote diarization annotation.
    Saves slices to temp_slices_dir using pii_safe_file_prefix in filenames.
    Returns a dictionary: {speaker_label: [{'path': slice_path, 'start': global_start_s, 'end': global_end_s}]}
    """
    logger.info(f"[Transcription Slicing] Slicing {audio_file_path} based on diarization (PII Prefix: {pii_safe_file_prefix}).")
    try:
        audio = AudioSegment.from_file(audio_file_path)
    except Exception as e:
        logger.error(f"[Transcription Slicing] Failed to load audio file {audio_file_path}: {e}")
        return None
    
    temp_slices_dir.mkdir(parents=True, exist_ok=True)
    
    speaker_segment_data = {} # {formatted_speaker_label: [segment_info_dicts]}
    segment_counter = 0

    for turn, _, speaker_label_pyannote in diarization_annotation.itertracks(yield_label=True):
        start_ms = int(turn.start * 1000)
        end_ms = int(turn.end * 1000)
        duration_ms = end_ms - start_ms

        if duration_ms < min_segment_duration_ms:
            logger.debug(f"Skipping very short segment for {speaker_label_pyannote} ({duration_ms}ms)")
            continue

        segment_audio = audio[start_ms:end_ms]
        
        # Format speaker label for directory/file naming (e.g., S0, S1)
        formatted_speaker_label = format_speaker_label_for_output(speaker_label_pyannote)

        speaker_specific_slice_dir = temp_slices_dir / formatted_speaker_label
        speaker_specific_slice_dir.mkdir(parents=True, exist_ok=True)
        
        slice_filename = f"{pii_safe_file_prefix}_{segment_counter:04d}_{formatted_speaker_label}_start{start_ms}_end{end_ms}.wav"
        slice_path = speaker_specific_slice_dir / slice_filename
        
        try:
            segment_audio.export(slice_path, format="wav")
        except Exception as e:
            logger.error(f"[Transcription Slicing] Failed to export audio slice {slice_path}: {e}")
            continue # Skip this slice

        if formatted_speaker_label not in speaker_segment_data:
            speaker_segment_data[formatted_speaker_label] = []
        
        speaker_segment_data[formatted_speaker_label].append({
            'path': str(slice_path),
            'start_s': turn.start, # Original start time in seconds from main audio
            'end_s': turn.end,     # Original end time in seconds from main audio
            'speaker': formatted_speaker_label # Store our formatted label
        })
        segment_counter += 1
        
    logger.info(f"[Transcription Slicing] Sliced audio into {segment_counter} segments across {len(speaker_segment_data)} speakers.")
    return speaker_segment_data


# Adapted from WhisperBite.whisperBite.transcribe_with_whisper
def transcribe_sliced_segments(
    whisper_model, 
    sliced_speaker_data: dict # Output from slice_audio_for_transcription
):
    """
    Transcribes pre-sliced audio segments using Whisper.
    Returns a flat list of transcribed segments with global timestamps.
    Each segment: {'text': str, 'start': float_sec, 'end': float_sec, 'speaker': str, 'language': str}
    """
    all_transcribed_segments = []
    
    for speaker_label, segments_info_list in sliced_speaker_data.items():
        logger.info(f"[Whisper Transcription] Transcribing {len(segments_info_list)} segments for speaker {speaker_label}...")
        for segment_info in segments_info_list:
            slice_path = segment_info['path']
            global_start_time_s = segment_info['start_s']
            # global_end_time_s = segment_info['end_s'] # Available if needed

            try:
                # Whisper transcribes the whole file, timestamps are relative to the start of the slice.
                transcription_result = whisper_model.transcribe(slice_path, word_timestamps=False) # Keep False for now
                
                text = transcription_result.get('text', "").strip()
                language = transcription_result.get('language', "unknown")

                if not text:
                    logger.debug(f"No text transcribed for slice: {slice_path}")
                    continue
                
                # Since Whisper's transcribe() on a short slice gives timestamps relative to slice start (usually 0),
                # and Whisper returns segments with their own start/end, we adjust these to be global.
                # For simplicity, if word_timestamps=False, Whisper gives a single 'text' for the whole slice.
                # The 'segments' key in result might still exist and have start/end for that text within the slice.
                # We'll take the full text of the slice and attribute it to the global start/end of that slice.
                
                # If Whisper's result for a slice provides segments, iterate them and adjust timestamps.
                # Otherwise, treat the whole slice transcription as one segment.
                if transcription_result.get('segments'):
                    for res_segment in transcription_result['segments']:
                        seg_text = res_segment['text'].strip()
                        if not seg_text:
                            continue
                        # Timestamps from Whisper segment are relative to the slice's start.
                        # Add the slice's global start time to make them global.
                        abs_seg_start = global_start_time_s + res_segment['start']
                        abs_seg_end = global_start_time_s + res_segment['end']
                        
                        all_transcribed_segments.append({
                            'text': seg_text,
                            'start': round(abs_seg_start, 3),
                            'end': round(abs_seg_end, 3),
                            'speaker': speaker_label,
                            'language': language
                        })
                elif text: # No segments, but there is text for the whole slice
                     all_transcribed_segments.append({
                        'text': text,
                        'start': round(global_start_time_s, 3),
                        # End time is the global end of the slice if Whisper doesn't give finer segment detail
                        'end': round(segment_info['end_s'], 3), 
                        'speaker': speaker_label,
                        'language': language
                    })

            except Exception as e:
                logger.error(f"[Whisper Transcription] Error transcribing slice {slice_path}: {e}")
                continue
                
    # Sort all segments by global start time
    all_transcribed_segments.sort(key=lambda x: x['start'])
    logger.info(f"[Whisper Transcription] Transcription complete. Generated {len(all_transcribed_segments)} text segments.")
    return all_transcribed_segments


def run_transcription(vocal_stem_path: str, diarization_file_path: str, output_dir_str: str, pii_safe_file_prefix: str, config: dict = None):
    """
    Transcribes an audio file using Whisper, guided by speaker diarization from an RTTM file.
    Uses pii_safe_file_prefix for naming the output transcript file.
    """
    if config is None:
        config = {}
    
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_slice_parent_dir = None # For try/finally cleanup

    logger.info(f"[Whisper Transcription] Starting for: {vocal_stem_path}")
    logger.info(f"[Whisper Transcription] Using RTTM: {diarization_file_path}")
    logger.info(f"[Whisper Transcription] PII-Safe Prefix for output: {pii_safe_file_prefix}")
    logger.info(f"[Whisper Transcription] Config: {config}")

    try:
        # 1. Load Diarization Annotation from RTTM
        if not Path(diarization_file_path).exists():
            logger.error(f"[Whisper Transcription] Diarization RTTM file not found: {diarization_file_path}")
            return None
        try:
            # We assume our RTTM's file_id matches the vocal_stem_path's stem.
            rttm_file_id = Path(vocal_stem_path).stem
            # diarization_annotation = Annotation.read_rttm(diarization_file_path, uri=rttm_file_id)
            # If uri doesn't work as expected (e.g. RTTM has no URI or different one), 
            # loading without uri and then filtering might be needed, or manual parsing.
            # For now, assume RTTM is specific to this one file or `uri` works.

            # New RTTM parsing logic
            rttm_data = RTTMParser().read(diarization_file_path)
            diarization_annotation = None
            
            # Attempt to get annotation by URI (file ID)
            if rttm_file_id in rttm_data.uri2annot:
                diarization_annotation = rttm_data.uri2annot[rttm_file_id]
                logger.info(f"[Whisper Transcription] Loaded RTTM annotation for URI: {rttm_file_id}")
            elif not rttm_data.uri2annot and rttm_data._uris:
                # If uri2annot is empty but there are URIs, it might mean the parser found URIs but no annotations for them.
                # More likely, if it's a single-file RTTM without explicit matching URI, try to get the first one if available.
                # This part is tricky without knowing the exact structure RTTMParser().read() gives for various RTTM files.
                # A safer approach if only one annotation is expected:
                annotations = list(rttm_data.iterannotations())
                if len(annotations) == 1:
                    diarization_annotation = annotations[0]
                    # Attempt to get the URI if possible, for logging or consistency
                    actual_uri = list(rttm_data.uris)[0] if rttm_data.uris else "unknown_uri"
                    logger.info(f"[Whisper Transcription] Loaded the single available RTTM annotation (URI found in RTTM: {actual_uri}, expected: {rttm_file_id}). Using this annotation.")
                elif len(annotations) > 1:
                    logger.warning(f"[Whisper Transcription] RTTM file {diarization_file_path} contains multiple annotations ({len(annotations)}) but none matched URI '{rttm_file_id}'. Cannot proceed without a specific annotation.")
                    return None
            # else: uri2annot might be empty and no _uris, or rttm_file_id just not found.

            if not diarization_annotation: # Check if annotation is empty
                 logger.error(f"[Whisper Transcription] Failed to load a valid diarization annotation from RTTM file {diarization_file_path} for expected file ID '{rttm_file_id}'.")
                 return None

        except Exception as e:
            logger.error(f"[Whisper Transcription] Failed to load/parse RTTM file {diarization_file_path} with RTTMParser: {e}")
            return None

        # 2. Setup for Slicing (temp directory)
        # Create a unique temporary directory for slices for this run
        temp_slice_parent_dir = Path(tempfile.mkdtemp(prefix="whisper_slices_", dir=output_dir))
        temp_slices_dir = temp_slice_parent_dir / "slices"
        
        # 3. Slice audio based on diarization
        sliced_speaker_data = slice_audio_for_transcription(
            vocal_stem_path, 
            diarization_annotation, 
            temp_slices_dir,
            pii_safe_file_prefix, # Pass the prefix down
            min_segment_duration_ms=config.get('min_slice_duration_ms', 500)
        )
        if not sliced_speaker_data:
            logger.error("[Whisper Transcription] Audio slicing failed or produced no segments.")
            return None

        # 4. Load Whisper Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        whisper_model_name = config.get("whisper_model_name", "base")
        logger.info(f"[Whisper Transcription] Loading Whisper model: {whisper_model_name} on device: {device}")
        try:
            whisper_model = whisper_load_model(whisper_model_name, device=device)
        except Exception as e:
            logger.error(f"[Whisper Transcription] Failed to load Whisper model '{whisper_model_name}': {e}")
            return None
            
        # 5. Transcribe Sliced Segments
        final_transcribed_segments = transcribe_sliced_segments(whisper_model, sliced_speaker_data)

        if not final_transcribed_segments:
            logger.warning("[Whisper Transcription] Transcription produced no segments.")
            # Create an empty transcript file to signify completion but no content
            final_text = ""
            # final_transcribed_segments remains empty list
        else:
            # Concatenate all text for a full transcript view (optional)
            final_text = " ".join([seg['text'] for seg in final_transcribed_segments])

        # 6. Format and Save Output JSON
        output_transcript_data = {
            "text": final_text,
            "segments": final_transcribed_segments,
            "language": final_transcribed_segments[0]['language'] if final_transcribed_segments else "unknown",
            "diarization_source_rttm": str(Path(diarization_file_path).resolve()),
            "whisper_model_used": whisper_model_name
        }
        
        transcript_filename = f"{pii_safe_file_prefix}_transcription.json" # Use PII-safe prefix directly
        transcript_file_path = output_dir / transcript_filename
        
        try:
            with open(transcript_file_path, 'w', encoding='utf-8') as f:
                json.dump(output_transcript_data, f, indent=4, ensure_ascii=False)
            logger.info(f"[Whisper Transcription] Transcript saved to: {transcript_file_path}")
        except Exception as e:
            logger.error(f"[Whisper Transcription] Failed to write transcript JSON: {e}")
            return None
            
        return {"transcript_json_path": str(transcript_file_path), "language": output_transcript_data["language"]}

    finally:
        # Cleanup temporary slice directory
        if temp_slice_parent_dir and temp_slice_parent_dir.exists():
            try:
                shutil.rmtree(temp_slice_parent_dir)
                logger.info(f"[Whisper Transcription] Cleaned up temporary slice directory: {temp_slice_parent_dir}")
            except Exception as e:
                logger.warning(f"[Whisper Transcription] Failed to cleanup temp slice directory {temp_slice_parent_dir}: {e}") 