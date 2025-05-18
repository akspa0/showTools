import logging
import os
import re # For sanitizing filenames
from pathlib import Path
import json
import torch
from whisper import load_model as whisper_load_model # Renamed to avoid conflict
from pyannote.core import Annotation, Segment
from pyannote.database.util import load_rttm # Correct import for RTTM loading
from pydub import AudioSegment # For slicing
import tempfile
import shutil # For cleaning up temp slice dir
from datetime import datetime

logger = logging.getLogger(__name__)

# --- Utility Functions ---

def _format_timestamp(seconds: float) -> str:
    """Converts seconds to HH:MM:SS.mmm format."""
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    seconds_val = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02d}:{minutes:02d}:{seconds_val:02d}.{milliseconds:03d}"

# Adapted from WhisperBite.utils - simplified for speaker labels like SPEAKER_00, S0 etc.
def format_speaker_label_for_output(label_from_pyannote: str) -> str:
    """Standardizes speaker labels from pyannote for output (e.g., 'SPEAKER_00' -> 'S0')."""
    if isinstance(label_from_pyannote, str):
        match = re.match(r"SPEAKER_(\d+)", label_from_pyannote)
        if match:
            return f"S{match.group(1)}"
    # Allow already formatted labels like S0, S1 to pass through
    if isinstance(label_from_pyannote, str) and re.match(r"S\d+", label_from_pyannote):
        return label_from_pyannote
    logger.warning(f"Could not format speaker label '{label_from_pyannote}', using original or converting to string.")
    return str(label_from_pyannote)


def _sanitize_filename(text: str, max_length: int = 40) -> str:
    """
    Sanitizes a string to be a valid filename component.
    Removes/replaces invalid characters, converts to lowercase,
    replaces spaces with underscores, and truncates.
    """
    if not text:
        return "untitled"
    sanitized = re.sub(r'[^a-zA-Z0-9_\\-]+', '_', text)
    sanitized = re.sub(r'[_\\-]+', '_', sanitized).strip('_')
    sanitized = sanitized.lower()
    return sanitized[:max_length].strip('_') or "segment" # Ensure not empty after stripping


# --- NEW Core Slicing Logic (inspired by WhisperBite) ---
def _perform_slicing_with_merging_and_fades(
    audio_file_path: str,
    diarization_annotation: Annotation,
    temp_output_dir_for_slices: Path,
    pii_safe_file_prefix: str, # For logging
    min_segment_duration_s: float = 0.5, # Min duration in seconds
    merge_gap_s: float = 0.4, # Max gap in seconds to merge segments of the same speaker
    fade_duration_ms: int = 50   # Fade in/out duration in ms
):
    """
    Slices audio based on diarization, merges close segments of the same speaker, applies fades,
    and saves temporary slices.
    Returns a dictionary: {formatted_speaker: [{'temp_slice_path': ..., 'start_time_orig': ..., 'end_time_orig': ..., 'sequence_id': ...}, ...]}
    """
    logger.info(f"[{pii_safe_file_prefix}] Starting audio slicing with merging and fades for: {audio_file_path}")
    logger.info(f"[{pii_safe_file_prefix}] Config: min_segment_duration_s={min_segment_duration_s}, merge_gap_s={merge_gap_s}, fade_duration_ms={fade_duration_ms}")

    try:
        audio = AudioSegment.from_file(audio_file_path)
        logger.debug(f"[{pii_safe_file_prefix}] Loaded audio for slicing: {audio_file_path}. Duration: {len(audio)}ms.")
    except FileNotFoundError:
        logger.error(f"[{pii_safe_file_prefix}] Audio file NOT FOUND for slicing: '{audio_file_path}'")
        return {}
    except Exception as e:
        logger.error(f"[{pii_safe_file_prefix}] Error loading audio file '{audio_file_path}' for slicing: {e}", exc_info=True)
        return {}

    # 1. Group initial segments by speaker from diarization_annotation
    raw_speaker_segments = {}
    raw_segment_count = 0
    for segment, _, label in diarization_annotation.itertracks(yield_label=True):
        raw_segment_count += 1
        duration_s = segment.end - segment.start
        if duration_s < min_segment_duration_s:
            logger.debug(f"[{pii_safe_file_prefix}] Skipping raw short segment for speaker {label} ({duration_s:.2f}s): {segment.start:.2f}s - {segment.end:.2f}s")
            continue
        
        formatted_speaker = format_speaker_label_for_output(label)
        if formatted_speaker not in raw_speaker_segments:
            raw_speaker_segments[formatted_speaker] = []
        raw_speaker_segments[formatted_speaker].append({'start': segment.start, 'end': segment.end, 'duration': duration_s})
    
    logger.info(f"[{pii_safe_file_prefix}] Initial processing: {raw_segment_count} raw segments from RTTM. After min_duration filter, collected segments for {len(raw_speaker_segments)} speakers.")

    # 2. Merge close segments for each speaker
    merged_speaker_segments = {}
    for speaker, segments in raw_speaker_segments.items():
        if not segments:
            continue
        segments.sort(key=lambda x: x['start']) # Sort by start time

        merged = []
        current_segment = segments[0]
        for next_segment in segments[1:]:
            gap = next_segment['start'] - current_segment['end']
            if gap < merge_gap_s: # If gap is small, merge
                current_segment['end'] = next_segment['end']
                current_segment['duration'] = current_segment['end'] - current_segment['start']
            else: # Gap is too large, start a new merged segment
                merged.append(current_segment)
                current_segment = next_segment
        merged.append(current_segment) # Add the last segment
        merged_speaker_segments[speaker] = merged
        logger.debug(f"[{pii_safe_file_prefix}] Speaker {speaker}: Merged {len(segments)} raw segments into {len(merged)} segments.")

    # 3. Export merged segments with fades to temporary files and collect info
    output_segment_info = {} # This will be like WhisperBite's segment_info
    global_sequence_counter = 0

    for speaker_label, segments in merged_speaker_segments.items():
        if not segments: continue

        speaker_temp_slice_dir = temp_output_dir_for_slices / speaker_label
        speaker_temp_slice_dir.mkdir(parents=True, exist_ok=True)
        
        output_segment_info[speaker_label] = []

        for i, seg_times in enumerate(segments):
            start_ms = seg_times['start'] * 1000
            end_ms = seg_times['end'] * 1000
            
            # Ensure duration is still above a threshold after potential merging (e.g. if min_segment_duration_s was very small)
            if (end_ms - start_ms) / 1000 < (min_segment_duration_s / 2): # Use a fraction of original min as sanity check
                 logger.debug(f"[{pii_safe_file_prefix}] Skipping very short segment for speaker {speaker_label} after merging ({(end_ms - start_ms)/1000:.2f}s).")
                 continue

            try:
                segment_audio = audio[start_ms:end_ms]
                
                # Apply fade in/out
                actual_fade_duration = min(fade_duration_ms, int(segment_audio.duration_seconds * 1000 / 4)) # Ensure fade is not too long
                if actual_fade_duration > 0: # Only apply if positive
                    segment_audio = segment_audio.fade_in(actual_fade_duration).fade_out(actual_fade_duration)

                # Temporary slice filename - simple, will be renamed later based on content
                temp_slice_filename = f"temp_slice_{speaker_label}_{global_sequence_counter:04d}.wav"
                temp_slice_path = speaker_temp_slice_dir / temp_slice_filename
                segment_audio.export(temp_slice_path, format="wav")

                output_segment_info[speaker_label].append({
                    'temp_slice_path': str(temp_slice_path),
                    'start_time_orig': seg_times['start'], # Original start time in seconds
                    'end_time_orig': seg_times['end'],     # Original end time in seconds
                    'sequence_id': global_sequence_counter
                })
                global_sequence_counter += 1
                logger.debug(f"[{pii_safe_file_prefix}] Exported temp slice: {temp_slice_path} for speaker {speaker_label} ({seg_times['start']:.2f}s - {seg_times['end']:.2f}s)")
            except Exception as e:
                logger.error(f"[{pii_safe_file_prefix}] Error exporting audio segment for speaker {speaker_label} ({seg_times['start']:.2f}s - {seg_times['end']:.2f}s): {e}", exc_info=True)

    total_temp_slices = sum(len(s) for s in output_segment_info.values())
    logger.info(f"[{pii_safe_file_prefix}] Slicing with merging and fades complete. Generated {total_temp_slices} temporary audio slices in '{temp_output_dir_for_slices}'.")
    return output_segment_info


# --- NEW Core Transcription Logic (inspired by WhisperBite) ---
def _transcribe_and_create_soundbites(
    whisper_model,
    segment_info_from_slicing: dict, # Output from _perform_slicing_...
    language_for_transcription: str | None,
    persistent_soundbite_output_dir: Path, # Main stage output dir (e.g., .../03_transcription/)
    pii_safe_file_prefix: str, # For logging
    min_words_for_filename: int = 5
):
    """
    Transcribes temporary audio slices, saves persistent .wav and .txt soundbites
    with PII-safe, content-derived names.
    Returns a list of transcribed segment data dictionaries for the main JSON.
    """
    all_final_transcribed_segments = []
    total_slices_to_transcribe = sum(len(s) for s in segment_info_from_slicing.values())
    logger.info(f"[{pii_safe_file_prefix}] Starting transcription of {total_slices_to_transcribe} audio slices and creating soundbites.")

    if not segment_info_from_slicing:
        logger.warning(f"[{pii_safe_file_prefix}] No segment info provided from slicing. Skipping transcription.")
        return []

    for speaker_label, segments_to_transcribe in segment_info_from_slicing.items():
        if not segments_to_transcribe: continue

        # Ensure speaker directory exists in the persistent output
        speaker_final_output_dir = persistent_soundbite_output_dir / speaker_label
        speaker_final_output_dir.mkdir(parents=True, exist_ok=True)

        for segment_data in segments_to_transcribe:
            temp_slice_path_str = segment_data['temp_slice_path']
            temp_slice_path = Path(temp_slice_path_str)
            sequence_id = segment_data['sequence_id']
            start_time_orig = segment_data['start_time_orig']
            end_time_orig = segment_data['end_time_orig']

            if not temp_slice_path.exists():
                logger.warning(f"[{pii_safe_file_prefix}] Temporary audio slice file not found, skipping: {temp_slice_path_str}")
                continue

            try:
                logger.debug(f"[{pii_safe_file_prefix}] Transcribing temp slice: {temp_slice_path_str} for speaker {speaker_label}")
                result = whisper_model.transcribe(str(temp_slice_path), language=language_for_transcription, verbose=False)
                unstripped_text = result['text']
                transcribed_text = unstripped_text.strip()
                
                logger.debug(f"[{pii_safe_file_prefix}] Slice {temp_slice_path_str} transcribed. Raw text: '{unstripped_text[:70]}...'. Stripped text: '{transcribed_text[:70]}...'")

                if not transcribed_text: # If transcription is empty, skip soundbite creation for this slice
                    logger.warning(f"[{pii_safe_file_prefix}] Empty transcription for slice {temp_slice_path_str} (speaker {speaker_label}). Skipping soundbite creation.")
                    continue

                # --- Soundbite Saving Logic ---
                first_words = "_".join(transcribed_text.split()[:min_words_for_filename])
                sanitized_first_words = _sanitize_filename(first_words) # _sanitize_filename ensures it's not empty

                soundbite_base_name = f"{sequence_id:04d}_{sanitized_first_words}"
                final_soundbite_audio_path = speaker_final_output_dir / f"{soundbite_base_name}.wav"
                final_soundbite_text_path = speaker_final_output_dir / f"{soundbite_base_name}.txt"

                # Copy the temporary audio slice to its new persistent soundbite path
                shutil.copy2(temp_slice_path, final_soundbite_audio_path)
                logger.debug(f"[{pii_safe_file_prefix}] Saved final soundbite audio: {final_soundbite_audio_path}")

                # Create the corresponding text file
                timestamp_str = f"{speaker_label} [{_format_timestamp(start_time_orig)} --> {_format_timestamp(end_time_orig)}]"
                with open(final_soundbite_text_path, 'w', encoding='utf-8') as f_text:
                    f_text.write(f"{timestamp_str}\n{transcribed_text}\n")
                logger.debug(f"[{pii_safe_file_prefix}] Saved final soundbite text: {final_soundbite_text_path}")
                
                all_final_transcribed_segments.append({
                    "speaker": speaker_label,
                    "start_time": start_time_orig,
                    "end_time": end_time_orig,
                    "text": transcribed_text,
                    "soundbite_audio_path": str(final_soundbite_audio_path.resolve()),
                    "soundbite_text_path": str(final_soundbite_text_path.resolve()),
                    "sequence_id": sequence_id # Retain for potential sorting/reference
                })

            except Exception as e:
                logger.error(f"[{pii_safe_file_prefix}] Error transcribing/saving soundbite for temp slice {temp_slice_path_str}: {e}", exc_info=True)
    
    # Sort final segments by original start time before returning
    all_final_transcribed_segments.sort(key=lambda x: x['start_time'])
    logger.info(f"[{pii_safe_file_prefix}] Transcription and soundbite creation complete. Processed {len(all_final_transcribed_segments)} final segments.")
    return all_final_transcribed_segments


# --- Main Module Function (Orchestrator) ---
def run_transcription(
    vocal_stem_path: str, 
    diarization_file_path: str, 
    output_dir_str: str, # This is the persistent output directory for the transcription stage
    pii_safe_file_prefix: str,
    config: dict = None
):
    """
    Performs transcription using Whisper (default) or Parakeet TDT 0.6B V2 (if asr_engine=='parakeet'), guided by Pyannote diarization.
    Outputs a JSON file containing the full transcript with speaker labels and timestamps.
    Also outputs individual .wav and .txt soundbites per speaker segment into speaker-specific subdirs.
    """
    asr_engine = 'whisper'
    if config and 'asr_engine' in config:
        asr_engine = config['asr_engine']
    logger.info(f"[{pii_safe_file_prefix}] Using ASR engine: {asr_engine}")
    if asr_engine == 'parakeet':
        try:
            from nemo.collections.asr.models import EncDecRNNTBPEModel
            class SimpleParakeetASR:
                def __init__(self, model_name="nvidia/parakeet-tdt-0.6b-v2", device=None):
                    self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
                    self.model = EncDecRNNTBPEModel.from_pretrained(model_name=model_name, map_location=self.device)
                    self.model.eval()
                def transcribe(self, audio_path):
                    # Returns a dict with 'text' and 'words' (if available)
                    result = self.model.transcribe([audio_path], batch_size=1, return_hypotheses=True)[0]
                    text = getattr(result, 'text', None)
                    words = getattr(result, 'words', None)
                    if words is not None and hasattr(words, '__iter__'):
                        words = [w if isinstance(w, dict) or isinstance(w, str) or not hasattr(w, '__dict__') else w.__dict__ for w in words]
                    return {"text": text, "words": words}
            asr_model = SimpleParakeetASR()
            logger.info(f"[{pii_safe_file_prefix}] Loaded SimpleParakeetASR model.")

            # --- Diarization-based segmenting and per-segment transcription ---
            persistent_stage_output_dir = Path(output_dir_str)
            persistent_stage_output_dir.mkdir(parents=True, exist_ok=True)
            # Validate RTTM
            if not diarization_file_path or not Path(diarization_file_path).exists():
                logger.error(f"[{pii_safe_file_prefix}] Diarization RTTM file not found: {diarization_file_path}")
                return {"error": "Diarization RTTM file not found", "transcript_json_path": None, "master_transcript_text_file": None, "soundbite_speaker_dirs": [], "soundbite_output_base_dir": None}
            rttm_data = load_rttm(diarization_file_path)
            if not rttm_data:
                logger.error(f"[{pii_safe_file_prefix}] RTTM file loaded empty: {diarization_file_path}")
                return {"error": "RTTM file loaded empty", "transcript_json_path": None, "master_transcript_text_file": None, "soundbite_speaker_dirs": [], "soundbite_output_base_dir": None}
            first_key = list(rttm_data.keys())[0]
            diarization_annotation = rttm_data[first_key]
            if not isinstance(diarization_annotation, Annotation):
                logger.error(f"[{pii_safe_file_prefix}] Invalid data type from RTTM (key: {first_key}): {type(diarization_annotation)}")
                return {"error": "Invalid data from RTTM", "transcript_json_path": None, "master_transcript_text_file": None, "soundbite_speaker_dirs": [], "soundbite_output_base_dir": None}
            logger.info(f"[{pii_safe_file_prefix}] RTTM loaded successfully (key: {first_key}).")

            # --- Slicing ---
            temp_dir = tempfile.TemporaryDirectory(prefix=f"{pii_safe_file_prefix}_parakeet_work_")
            temp_slices_subdir = Path(temp_dir.name) / "initial_audio_slices"
            temp_slices_subdir.mkdir(parents=True, exist_ok=True)
            segment_info_for_transcription = _perform_slicing_with_merging_and_fades(
                audio_file_path=vocal_stem_path,
                diarization_annotation=diarization_annotation,
                temp_output_dir_for_slices=temp_slices_subdir,
                pii_safe_file_prefix=pii_safe_file_prefix,
                min_segment_duration_s=config.get("min_segment_duration_s", 0.5),
                merge_gap_s=config.get("merge_gap_s", 0.4),
                fade_duration_ms=config.get("fade_duration_ms", 50)
            )
            final_transcribed_segments = []
            for speaker_label, segments in segment_info_for_transcription.items():
                if not segments: continue
                speaker_final_output_dir = Path(output_dir_str) / speaker_label
                speaker_final_output_dir.mkdir(parents=True, exist_ok=True)
                for segment_data in segments:
                    temp_slice_path_str = segment_data['temp_slice_path']
                    temp_slice_path = Path(temp_slice_path_str)
                    sequence_id = segment_data['sequence_id']
                    start_time_orig = segment_data['start_time_orig']
                    end_time_orig = segment_data['end_time_orig']
                    if not temp_slice_path.exists():
                        logger.warning(f"[{pii_safe_file_prefix}] Temporary audio slice file not found, skipping: {temp_slice_path_str}")
                        continue
                    try:
                        result = asr_model.transcribe(str(temp_slice_path))
                        transcribed_text = (result.get('text') or '').strip()
                        words = result.get('words', [])
                        # TXT: [start_time --> end_time]\ntext
                        soundbite_base_name = f"{sequence_id:04d}_{_sanitize_filename(' '.join(transcribed_text.split()[:config.get('min_words_for_filename', 5)]))}"
                        final_soundbite_audio_path = speaker_final_output_dir / f"{soundbite_base_name}.wav"
                        final_soundbite_text_path = speaker_final_output_dir / f"{soundbite_base_name}.txt"
                        final_soundbite_json_path = speaker_final_output_dir / f"{soundbite_base_name}.json"
                        shutil.copy2(temp_slice_path, final_soundbite_audio_path)
                        timestamp_str = f"{speaker_label} [{_format_timestamp(start_time_orig)} --> {_format_timestamp(end_time_orig)}]"
                        with open(final_soundbite_text_path, 'w', encoding='utf-8') as f_text:
                            f_text.write(f"{timestamp_str}\n{transcribed_text}\n")
                        with open(final_soundbite_json_path, 'w', encoding='utf-8') as f_json:
                            json.dump({
                                "speaker": speaker_label,
                                "start_time": start_time_orig,
                                "end_time": end_time_orig,
                                "text": transcribed_text,
                                "words": words,
                                "soundbite_audio_path": str(final_soundbite_audio_path.resolve()),
                                "soundbite_text_path": str(final_soundbite_text_path.resolve()),
                                "sequence_id": sequence_id
                            }, f_json, indent=2, ensure_ascii=False)
                        final_transcribed_segments.append({
                            "speaker": speaker_label,
                            "start_time": start_time_orig,
                            "end_time": end_time_orig,
                            "text": transcribed_text,
                            "words": words,
                            "soundbite_audio_path": str(final_soundbite_audio_path.resolve()),
                            "soundbite_text_path": str(final_soundbite_text_path.resolve()),
                            "sequence_id": sequence_id
                        })
                    except Exception as e:
                        logger.error(f"[{pii_safe_file_prefix}] Error transcribing/saving soundbite for temp slice {temp_slice_path_str}: {e}", exc_info=True)
            # Master outputs
            output_transcript_json_path = Path(output_dir_str) / f"{pii_safe_file_prefix}_transcription.json"
            with open(output_transcript_json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "file_info": {
                        "original_audio_path": str(vocal_stem_path),
                        "rttm_file_path": str(diarization_file_path),
                        "processing_date": datetime.now().isoformat(),
                        "asr_engine": "parakeet"
                    },
                    "segments": final_transcribed_segments
                }, f, indent=4, ensure_ascii=False)
            master_transcript_txt_path = Path(output_dir_str) / f"{pii_safe_file_prefix}_master_transcript.txt"
            with open(master_transcript_txt_path, 'w', encoding='utf-8') as f_txt:
                for seg in final_transcribed_segments:
                    speaker = seg.get('speaker', 'UNK')
                    start = seg.get('start_time', 0.0)
                    end = seg.get('end_time', 0.0)
                    text = seg.get('text', '').strip()
                    if text:
                        f_txt.write(f"{speaker} [{_format_timestamp(start)} --> {_format_timestamp(end)}]: {text}\n")
            temp_dir.cleanup()
            soundbite_dirs = sorted(list(set([
                str(Path(seg['soundbite_audio_path']).parent)
                for seg in final_transcribed_segments if 'soundbite_audio_path' in seg
            ])))
            return {
                "transcript_json_path": str(output_transcript_json_path) if output_transcript_json_path.exists() else None,
                "master_transcript_text_file": str(master_transcript_txt_path) if master_transcript_txt_path.exists() else None,
                "transcript_file": str(output_transcript_json_path) if output_transcript_json_path.exists() else None,
                "soundbite_speaker_dirs": soundbite_dirs,
                "soundbite_output_base_dir": str(Path(output_dir_str))
            }
        except Exception as e:
            logger.error(f"[{pii_safe_file_prefix}] Parakeet ASR failed: {e}")
            return {'transcript_json_path': None, 'master_transcript_text_file': None, 'transcript_file': None, 'soundbite_speaker_dirs': [], 'soundbite_output_base_dir': None}
    # Default: Whisper
    if config is None: config = {}
    
    logger.info(f"[{pii_safe_file_prefix}] Starting transcription (WhisperBite-style) for: '{vocal_stem_path}', RTTM: '{diarization_file_path}'")
    persistent_stage_output_dir = Path(output_dir_str)
    persistent_stage_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Configuration ---
    whisper_model_name = config.get("whisper_model_name", "base.en")
    language = config.get("language_for_transcription", None)
    min_segment_duration_s = config.get("min_segment_duration_s", 0.5) # from previous default of 500ms
    merge_gap_s = config.get("merge_gap_s", 0.4)
    fade_duration_ms = config.get("fade_duration_ms", 50)
    min_words_for_filename = config.get("min_words_for_filename", 5)


    # --- Validate Inputs ---
    if not vocal_stem_path or not Path(vocal_stem_path).exists():
        logger.error(f"[{pii_safe_file_prefix}] Vocal stem file not found: {vocal_stem_path}")
        return {"error": "Vocal stem file not found", "transcript_json_path": None, "master_transcript_txt_path": None, "soundbite_speaker_dirs": [], "soundbite_output_base_dir": None}
    if not diarization_file_path or not Path(diarization_file_path).exists():
        logger.error(f"[{pii_safe_file_prefix}] Diarization RTTM file not found: {diarization_file_path}")
        return {"error": "Diarization RTTM file not found", "transcript_json_path": None, "master_transcript_txt_path": None, "soundbite_speaker_dirs": [], "soundbite_output_base_dir": None}

    # --- Load Whisper Model ---
    try:
        logger.info(f"[{pii_safe_file_prefix}] Loading Whisper model: {whisper_model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[{pii_safe_file_prefix}] Whisper using device: {device}")
        whisper_model = whisper_load_model(whisper_model_name, device=device)
    except Exception as e:
        logger.error(f"[{pii_safe_file_prefix}] Failed to load Whisper model '{whisper_model_name}': {e}", exc_info=True)
        return {"error": f"Failed to load Whisper model: {e}", "transcript_json_path": None, "master_transcript_txt_path": None, "soundbite_speaker_dirs": [], "soundbite_output_base_dir": None}

    # --- Load Diarization (RTTM) ---
    logger.info(f"[{pii_safe_file_prefix}] Loading RTTM: {diarization_file_path}")
    try:
        rttm_data = load_rttm(diarization_file_path)
        if not rttm_data:
            logger.error(f"[{pii_safe_file_prefix}] RTTM file loaded empty: {diarization_file_path}")
            return {"error": "RTTM file loaded empty", "transcript_json_path": None, "master_transcript_txt_path": None, "soundbite_speaker_dirs": [], "soundbite_output_base_dir": None}
        first_key = list(rttm_data.keys())[0]
        diarization_annotation = rttm_data[first_key]
        if not isinstance(diarization_annotation, Annotation):
            logger.error(f"[{pii_safe_file_prefix}] Invalid data type from RTTM (key: {first_key}): {type(diarization_annotation)}")
            return {"error": "Invalid data from RTTM", "transcript_json_path": None, "master_transcript_txt_path": None, "soundbite_speaker_dirs": [], "soundbite_output_base_dir": None}
        logger.info(f"[{pii_safe_file_prefix}] RTTM loaded successfully (key: {first_key}).")
    except Exception as e:
        logger.error(f"[{pii_safe_file_prefix}] Error loading RTTM '{diarization_file_path}': {e}", exc_info=True)
        return {"error": f"Error loading RTTM: {e}", "transcript_json_path": None, "master_transcript_txt_path": None, "soundbite_speaker_dirs": [], "soundbite_output_base_dir": None}

    # --- Main Processing using a temporary directory for intermediate files ---
    final_transcribed_segments = []
    # Use a single encompassing temp dir for all slices created by this run_transcription call
    with tempfile.TemporaryDirectory(prefix=f"{pii_safe_file_prefix}_transcription_work_") as main_temp_dir_str:
        main_temp_dir = Path(main_temp_dir_str)
        logger.info(f"[{pii_safe_file_prefix}] Created main temporary working directory: {main_temp_dir}")
        
        # This sub-directory will hold the first pass of slices (merged, faded, but not yet transcribed/renamed)
        temp_slices_subdir = main_temp_dir / "initial_audio_slices"
        temp_slices_subdir.mkdir(parents=True, exist_ok=True)

        # 1. Perform slicing with merging and fades (saves to temp_slices_subdir)
        segment_info_for_transcription = _perform_slicing_with_merging_and_fades(
            audio_file_path=vocal_stem_path,
            diarization_annotation=diarization_annotation,
            temp_output_dir_for_slices=temp_slices_subdir,
            pii_safe_file_prefix=pii_safe_file_prefix,
            min_segment_duration_s=min_segment_duration_s,
            merge_gap_s=merge_gap_s,
            fade_duration_ms=fade_duration_ms
        )

        if not segment_info_for_transcription or not any(segment_info_for_transcription.values()):
            logger.warning(f"[{pii_safe_file_prefix}] No processable audio slices were generated after slicing/merging. Transcription will be empty.")
            # Proceed to create an empty JSON, but the list will be empty.
        else:
            # 2. Transcribe slices and create final soundbites (saves to persistent_stage_output_dir)
            final_transcribed_segments = _transcribe_and_create_soundbites(
                whisper_model=whisper_model,
                segment_info_from_slicing=segment_info_for_transcription,
                language_for_transcription=language,
                persistent_soundbite_output_dir=persistent_stage_output_dir, # Final destination for S0/, S1/ etc.
                pii_safe_file_prefix=pii_safe_file_prefix,
                min_words_for_filename=min_words_for_filename
            )
        
        # main_temp_dir (and temp_slices_subdir within it) will be cleaned up automatically when 'with' block exits.
        logger.info(f"[{pii_safe_file_prefix}] Temporary working directory {main_temp_dir} will be cleaned up.")


    # --- Final Output JSON ---
    output_transcript_json_path = persistent_stage_output_dir / f"{pii_safe_file_prefix}_transcription.json"
    try:
        with open(output_transcript_json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "file_info": {
                    "original_audio_path": str(vocal_stem_path),
                    "rttm_file_path": str(diarization_file_path),
                    "processing_date": datetime.now().isoformat(),
                    "whisper_model": whisper_model_name
                },
                "segments": final_transcribed_segments
            }, f, indent=4, ensure_ascii=False)
        logger.info(f"[{pii_safe_file_prefix}] Main transcript JSON saved to: {output_transcript_json_path}")
    except Exception as e:
        logger.error(f"[{pii_safe_file_prefix}] Error writing main transcript JSON {output_transcript_json_path}: {e}")
        # Decide if we should return early or try to continue with other outputs

    # --- Write the master_transcript.txt file ---
    master_transcript_txt_path = persistent_stage_output_dir / f"{pii_safe_file_prefix}_master_transcript.txt"
    try:
        with open(master_transcript_txt_path, 'w', encoding='utf-8') as f_txt:
            for segment_data in final_transcribed_segments:
                speaker = segment_data.get('speaker', 'UNK')
                start = segment_data.get('start_time', 0.0)
                end = segment_data.get('end_time', 0.0)
                text = segment_data.get('text', '').strip()
                if text:
                    f_txt.write(f"{speaker} [{_format_timestamp(start)} --> {_format_timestamp(end)}]: {text}\n")
        logger.info(f"[{pii_safe_file_prefix}] Master transcript text file saved to: {master_transcript_txt_path}")
    except Exception as e:
        logger.error(f"[{pii_safe_file_prefix}] Error writing master transcript text file {master_transcript_txt_path}: {e}")
        master_transcript_txt_path = None # Indicate failure

    # --- Cleanup temporary working directory ---
    if main_temp_dir.exists():
        shutil.rmtree(main_temp_dir)

    # --- Final Output ---
    soundbite_dirs = []
    if final_transcribed_segments: # Only try to get parent dirs if segments exist
        soundbite_dirs = sorted(list(set([
            str(Path(seg['soundbite_audio_path']).parent) 
            for seg in final_transcribed_segments if 'soundbite_audio_path' in seg
        ])))
    
    return {
        "transcript_json_path": str(output_transcript_json_path) if output_transcript_json_path.exists() else None,
        "master_transcript_text_file": str(master_transcript_txt_path) if master_transcript_txt_path and master_transcript_txt_path.exists() else None,
        "transcript_file": str(output_transcript_json_path) if output_transcript_json_path.exists() else None,
        "soundbite_speaker_dirs": soundbite_dirs,
        "soundbite_output_base_dir": str(persistent_stage_output_dir) # For call_processor to locate speaker folders
    }


# --- Main block for testing (remains similar but now tests the refactored logic) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] - %(name)s.%(funcName)s - %(message)s')
    logger.info("Starting transcription_module.py test run (refactored WhisperBite style)...")

    test_output_base_dir = Path("workspace/transcription_module_test_outputs_wb_style")
    test_output_base_dir.mkdir(parents=True, exist_ok=True)
    test_pii_prefix = "call_20240102-140000_test_wb"

    dummy_vocal_stem_path = test_output_base_dir / f"{test_pii_prefix}_vocals_normalized.wav"
    if not dummy_vocal_stem_path.exists():
        try:
            silence = AudioSegment.silent(duration=5000, frame_rate=16000) # 5 secs
            # Add a couple of beeps to make it non-empty for transcription
            beep1 = AudioSegment.sine(440).to_mono()[:200] # 200ms beep
            beep2 = AudioSegment.sine(880).to_mono()[:200]
            silence = silence.overlay(beep1, position=1000)
            silence = silence.overlay(beep2, position=3000)
            silence.export(dummy_vocal_stem_path, format="wav")
            logger.info(f"Created dummy vocal stem with beeps: {dummy_vocal_stem_path}")
        except Exception as e:
            logger.error(f"Could not create dummy vocal stem: {e}. Test may fail.", exc_info=True)
            
    dummy_rttm_path = test_output_base_dir / f"{test_pii_prefix}_vocals_normalized_diarization.rttm"
    if not dummy_rttm_path.exists():
        # RTTM for the 5-second audio with two speakers and some silences
        # File ID in RTTM (second field) should match the stem of the audio file it refers to.
        rttm_file_id_stem = Path(dummy_vocal_stem_path).stem 
        rttm_content = (
            f"SPEAKER {rttm_file_id_stem} 1 0.800 0.600 <NA> <NA> SPEAKER_00 <NA> <NA>\\n"  # Corresponds to beep1 + surrounding
            f"SPEAKER {rttm_file_id_stem} 1 1.400 1.000 <NA> <NA> SPEAKER_01 <NA> <NA>\\n"  # Some speech for S1
            f"SPEAKER {rttm_file_id_stem} 1 2.800 0.700 <NA> <NA> SPEAKER_00 <NA> <NA>\\n"  # Corresponds to beep2 + surrounding
        )
        with open(dummy_rttm_path, 'w') as f_rttm:
            f_rttm.write(rttm_content)
        logger.info(f"Created dummy RTTM: {dummy_rttm_path}")

    current_test_output_dir = test_output_base_dir / "run_01_transcription_output"
    current_test_output_dir.mkdir(parents=True, exist_ok=True)
    
    test_config = {
        "whisper_model_name": "tiny.en",
        "language_for_transcription": "en",
        "min_segment_duration_s": 0.2, # Allow short segments for test beeps
        "merge_gap_s": 0.1,          # Small gap for merging
        "fade_duration_ms": 20,
        "min_words_for_filename": 2
    }

    logger.info(f"--- Test Parameters ---")
    logger.info(f"Vocal Stem: {dummy_vocal_stem_path}")
    logger.info(f"RTTM File: {dummy_rttm_path}")
    logger.info(f"Output Dir: {current_test_output_dir}")
    logger.info(f"PII Prefix: {test_pii_prefix}")
    logger.info(f"Config: {json.dumps(test_config, indent=2)}")
    logger.info(f"----------------------")

    if not dummy_vocal_stem_path.exists() or not dummy_rttm_path.exists():
        logger.error("Test requires dummy_vocal_stem_path and dummy_rttm_path to exist. Please create them or point to valid files.")
    else:
        result = run_transcription(
            vocal_stem_path=str(dummy_vocal_stem_path),
            diarization_file_path=str(dummy_rttm_path),
            output_dir_str=str(current_test_output_dir),
            pii_safe_file_prefix=test_pii_prefix,
            config=test_config
        )

        logger.info("--- Test Run Transcription Result ---")
        if result and result.get("transcript_json_path"):
            logger.info(f"Message: {result.get('message')}")
            logger.info(f"Transcription main JSON: {result['transcript_json_path']}")
            logger.info(f"Soundbite speaker dirs: {result.get('soundbite_speaker_dirs')}")
            logger.info(f"Please inspect the contents of: {current_test_output_dir}")
        else:
            logger.error(f"Test run_transcription failed or returned unexpected result: {result}")

    logger.info("Transcription_module.py (WhisperBite style) test run finished.") 