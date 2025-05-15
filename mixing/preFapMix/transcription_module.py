import logging
import os
import re # For sanitizing filenames
from pathlib import Path
import json
import torch
from whisper import load_model as whisper_load_model # Renamed to avoid conflict
from pyannote.core import Annotation
from pydub import AudioSegment # For slicing
import tempfile
import shutil # For cleaning up temp slice dir

logger = logging.getLogger(__name__)

# --- Utility Functions ---

def _format_timestamp(seconds: float) -> str:
    """Converts seconds to HH:MM:SS.mmm format."""
    # ... (implementation as previously defined or standard library if available) ...
    # For now, a basic version:
    # This is a simplified version from whisper.utils.format_timestamp
    # Consider using whisper.utils.format_timestamp directly if robust formatting is needed
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
    return label_from_pyannote # Fallback


def _sanitize_filename(text: str, max_length: int = 40) -> str:
    """
    Sanitizes a string to be a valid filename component.
    Removes/replaces invalid characters, converts to lowercase,
    replaces spaces with underscores, and truncates.
    """
    if not text:
        return "untitled"
    # Remove or replace characters not suitable for filenames
    # Keep alphanumeric, underscore, hyphen. Replace others.
    sanitized = re.sub(r'[^a-zA-Z0-9_\\-]+', '_', text)
    # Replace multiple underscores/hyphens with a single one
    sanitized = re.sub(r'[_\\-]+', '_', sanitized)
    # Lowercase
    sanitized = sanitized.lower()
    # Truncate
    return sanitized[:max_length].strip('_')


# Adapted from WhisperBite.whisperBite.slice_audio_by_speaker
def slice_audio_for_transcription(
    audio_file_path: str, 
    diarization_annotation: Annotation, 
    temp_slices_dir: Path,
    pii_safe_file_prefix: str, # For logging context
    min_segment_duration_ms: int = 500 # Min duration in ms to avoid tiny slices
):
    """
    Slices an audio file based on pyannote diarization annotation.
    Saves slices to temp_slices_dir.
    Returns a list of dictionaries, each representing a slice with its path, speaker, start, end, and a new sequence_id.
    """
    logger.info(f"[{pii_safe_file_prefix}] Slicing audio '{audio_file_path}' based on diarization. Min segment: {min_segment_duration_ms}ms.")
    try:
        audio = AudioSegment.from_file(audio_file_path)
        logger.debug(f"[{pii_safe_file_prefix}] Loaded audio for slicing: {audio_file_path}. Duration: {len(audio)}ms.")
    except Exception as e:
        logger.error(f"[{pii_safe_file_prefix}] Error loading audio file '{audio_file_path}' for slicing: {e}")
        return []

    speaker_audio_segments = []
    processed_segments_count = 0
    sequence_id_counter = 0 # Initialize sequence ID counter

    for segment, track, label in diarization_annotation.itertracks(yield_label=True):
        start_ms = segment.start * 1000
        end_ms = segment.end * 1000
        duration_ms = end_ms - start_ms

        if duration_ms < min_segment_duration_ms:
            logger.debug(f"[{pii_safe_file_prefix}] Skipping short segment for speaker {label} ({duration_ms}ms): {segment.start:.2f}s - {segment.end:.2f}s")
            continue

        speaker_label_formatted = format_speaker_label_for_output(label)
        speaker_slice_dir = temp_slices_dir / speaker_label_formatted
        speaker_slice_dir.mkdir(parents=True, exist_ok=True)

        # Use a temporary, simple name for the slice file. It will be renamed later.
        # Include sequence_id here for uniqueness if many small segments from one speaker.
        slice_filename = f"{pii_safe_file_prefix}_{speaker_label_formatted}_seg_{processed_segments_count:04d}_temp.wav"
        slice_path = speaker_slice_dir / slice_filename
        
        try:
            segment_audio = audio[start_ms:end_ms]
            segment_audio.export(slice_path, format="wav")
            
            speaker_audio_segments.append({
                'temp_slice_path': str(slice_path), 
                'speaker': speaker_label_formatted,
                'start_time_orig': segment.start, # Original start time in seconds from diarization
                'end_time_orig': segment.end,     # Original end time in seconds from diarization
                'sequence_id': sequence_id_counter # Add sequence ID
            })
            sequence_id_counter += 1 # Increment for next segment
            processed_segments_count += 1
            logger.debug(f"[{pii_safe_file_prefix}] Exported slice: {slice_path} for speaker {speaker_label_formatted} ({segment.start:.2f}s - {segment.end:.2f}s)")
        except Exception as e:
            logger.error(f"[{pii_safe_file_prefix}] Error exporting audio segment for speaker {speaker_label_formatted} ({segment.start:.2f}s - {segment.end:.2f}s): {e}")
            # Continue to next segment if one slice fails

    logger.info(f"[{pii_safe_file_prefix}] Slicing complete. Generated {len(speaker_audio_segments)} temporary audio slices in '{temp_slices_dir}'.")
    return speaker_audio_segments


def transcribe_sliced_segments(
    whisper_model, 
    sliced_audio_segments: list, 
    language_for_transcription: str | None, # Can be None for auto-detect
    pii_safe_file_prefix: str,
    persistent_output_dir: Path # Main output dir for the transcription stage
):
    """
    Transcribes a list of audio slices using the provided Whisper model.
    Saves individual .wav and .txt soundbites to persistent_output_dir / speaker_label /.
    Updates segment dictionaries with paths to these soundbites.
    Returns a list of transcribed segments (dictionaries).
    """
    all_transcribed_segments = []
    logger.info(f"[{pii_safe_file_prefix}] Starting transcription of {len(sliced_audio_segments)} audio slices.")

    for segment_info in sliced_audio_segments:
        temp_slice_path_str = segment_info['temp_slice_path']
        temp_slice_path = Path(temp_slice_path_str)
        speaker_label = segment_info['speaker']
        sequence_id = segment_info['sequence_id']
        start_time_orig = segment_info['start_time_orig']
        end_time_orig = segment_info['end_time_orig']

        if not temp_slice_path.exists():
            logger.warning(f"[{pii_safe_file_prefix}] Audio slice file not found, skipping: {temp_slice_path_str}")
            continue

        try:
            logger.debug(f"[{pii_safe_file_prefix}] Transcribing slice: {temp_slice_path_str} for speaker {speaker_label}")
            # The result object contains the recognized text, segments, and language information.
            # For a single slice, we typically expect one dominant segment in 'result['segments']'
            # or just the full text in 'result['text']'.
            # We pass language=None to let Whisper auto-detect if not specified.
            result = whisper_model.transcribe(temp_slice_path_str, language=language_for_transcription, verbose=False) # verbose=False or True for more Whisper output
            transcribed_text = result['text'].strip()
            
            logger.debug(f"[{pii_safe_file_prefix}] Slice {temp_slice_path_str} transcribed. Text: '{transcribed_text[:50]}...'")

            # --- Soundbite Saving Logic ---
            speaker_output_dir = persistent_output_dir / speaker_label
            speaker_output_dir.mkdir(parents=True, exist_ok=True)

            # Create filename from first few words
            first_words = "_".join(transcribed_text.split()[:5]) # Take first 5 words
            sanitized_first_words = _sanitize_filename(first_words)
            if not sanitized_first_words or sanitized_first_words == "_": # Handle empty or only underscore
                sanitized_first_words = "segment"

            soundbite_base_name = f"{sequence_id:04d}_{sanitized_first_words}"
            
            soundbite_audio_path = speaker_output_dir / f"{soundbite_base_name}.wav"
            soundbite_text_path = speaker_output_dir / f"{soundbite_base_name}.txt"

            # Copy the temporary audio slice to its new persistent soundbite path
            shutil.copy2(temp_slice_path, soundbite_audio_path)
            logger.debug(f"[{pii_safe_file_prefix}] Saved soundbite audio: {soundbite_audio_path}")

            # Create the corresponding text file with timestamp and transcription
            timestamp_str = f"[{_format_timestamp(start_time_orig)} --> {_format_timestamp(end_time_orig)}]"
            with open(soundbite_text_path, 'w', encoding='utf-8') as f_text:
                f_text.write(f"{timestamp_str}\\n")
                f_text.write(transcribed_text + "\\n")
            logger.debug(f"[{pii_safe_file_prefix}] Saved soundbite text: {soundbite_text_path}")
            
            # Prepare segment data for the main JSON output
            # Note: Whisper's result['segments'] for a short slice might be simple.
            # We are creating a single segment entry based on the overall transcription of the slice.
            # If Whisper provided detailed segments for the slice, one might iterate through result['segments']
            # and adjust start/end times relative to the slice's own start, then add to original start_time_orig.
            # For simplicity here, we treat the whole slice transcription as one segment.
            segment_output_data = {
                "speaker": speaker_label,
                "start_time": start_time_orig, # Original start time from diarization
                "end_time": end_time_orig,     # Original end time from diarization
                "text": transcribed_text,
                "soundbite_audio_path": str(soundbite_audio_path.resolve()),
                "soundbite_text_path": str(soundbite_text_path.resolve())
                # Include other Whisper segment details if needed (e.g., tokens, confidence) by parsing `result` more deeply.
            }
            all_transcribed_segments.append(segment_output_data)

        except Exception as e:
            logger.error(f"[{pii_safe_file_prefix}] Error transcribing audio slice {temp_slice_path_str}: {e}", exc_info=True)
            # Optionally, add a placeholder or skip this segment in the final output.
            # For now, we just log and continue.

    logger.info(f"[{pii_safe_file_prefix}] Transcription of slices complete. Processed {len(all_transcribed_segments)} segments with soundbites.")
    return all_transcribed_segments


# --- Main Module Function ---
def run_transcription(
    vocal_stem_path: str, 
    diarization_file_path: str, 
    output_dir_str: str, 
    pii_safe_file_prefix: str, # Added for PII-safe output naming
    config: dict = None
):
    """
    Performs transcription using Whisper, guided by Pyannote diarization.
    Outputs a JSON file containing the full transcript with speaker labels and timestamps.
    Also outputs individual .wav and .txt soundbites per speaker segment into speaker-specific subdirs.
    """
    if config is None: config = {}
    
    logger.info(f"[{pii_safe_file_prefix}] Starting transcription process for vocal stem: '{vocal_stem_path}' using diarization: '{diarization_file_path}'")
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Configuration ---
    whisper_model_name = config.get("whisper_model_name", "base.en") # e.g., "base.en", "small", "medium"
    language_for_transcription = config.get("language_for_transcription", None) # None for auto-detect, or "en", "es", etc.
    min_segment_duration_ms_slicing = config.get("min_segment_duration_ms_slicing", 500)

    # --- Validate Inputs ---
    if not vocal_stem_path or not Path(vocal_stem_path).exists():
        logger.error(f"[{pii_safe_file_prefix}] Vocal stem file not found or path is None: {vocal_stem_path}")
        return {"error": "Vocal stem file not found", "transcript_json_path": None, "soundbite_dirs_info": None}
    if not diarization_file_path or not Path(diarization_file_path).exists():
        logger.error(f"[{pii_safe_file_prefix}] Diarization RTTM file not found or path is None: {diarization_file_path}")
        return {"error": "Diarization RTTM file not found", "transcript_json_path": None, "soundbite_dirs_info": None}

    # --- Load Whisper Model ---
    try:
        logger.info(f"[{pii_safe_file_prefix}] Loading Whisper model: {whisper_model_name}")
        # Check if CUDA is available and use it if so, for faster transcription
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[{pii_safe_file_prefix}] Whisper will use device: {device}")
        whisper_model = whisper_load_model(whisper_model_name, device=device)
        logger.info(f"[{pii_safe_file_prefix}] Whisper model '{whisper_model_name}' loaded successfully on {device}.")
    except Exception as e:
        logger.error(f"[{pii_safe_file_prefix}] Failed to load Whisper model '{whisper_model_name}': {e}", exc_info=True)
        return {"error": f"Failed to load Whisper model: {e}", "transcript_json_path": None, "soundbite_dirs_info": None}

    # --- Load Diarization ---
    try:
        logger.debug(f"[{pii_safe_file_prefix}] Loading diarization from RTTM: {diarization_file_path}")
        # diarization_annotation = RTTMParser().read(diarization_file_path) # OLD way causing import error
        diarization_annotation = Annotation.read_rttm(diarization_file_path) # CORRECTED way
        logger.info(f"[{pii_safe_file_prefix}] Successfully loaded diarization data from {diarization_file_path}. Found {len(diarization_annotation.get_timeline().support())} speech segments for {len(diarization_annotation.labels())} speakers.")
    except Exception as e:
        logger.error(f"[{pii_safe_file_prefix}] Error loading or parsing RTTM diarization file '{diarization_file_path}': {e}", exc_info=True)
        return {"error": f"Error loading RTTM: {e}", "transcript_json_path": None, "soundbite_dirs_info": None}

    # --- Slicing and Transcription ---
    # Create a temporary directory for audio slices
    with tempfile.TemporaryDirectory(prefix=f"{pii_safe_file_prefix}_slices_") as temp_slices_dir_str:
        temp_slices_dir = Path(temp_slices_dir_str)
        logger.info(f"[{pii_safe_file_prefix}] Created temporary directory for audio slices: {temp_slices_dir}")

        # 1. Slice audio based on diarization
        sliced_segments_info = slice_audio_for_transcription(
            vocal_stem_path, 
            diarization_annotation, 
            temp_slices_dir,
            pii_safe_file_prefix,
            min_segment_duration_ms=min_segment_duration_ms_slicing
        )

        if not sliced_segments_info:
            logger.warning(f"[{pii_safe_file_prefix}] No audio slices were generated. Transcription cannot proceed.")
            # Cleanup of temp_slices_dir happens automatically due to 'with' statement
            return {"error": "No audio slices generated from diarization.", "transcript_json_path": None, "soundbite_dirs_info": None}

        # 2. Transcribe sliced segments and save soundbites
        # The 'output_dir' passed here is the main output directory for this transcription stage,
        # where speaker subdirectories and soundbites will be saved.
        transcribed_segments_data = transcribe_sliced_segments(
            whisper_model, 
            sliced_segments_info, 
            language_for_transcription,
            pii_safe_file_prefix,
            persistent_output_dir=output_dir # Main stage output dir
        )
        
        # transcribed_segments_data now contains paths to persistent soundbites

    # --- Final Output ---
    # Main JSON transcript file (using pii_safe_file_prefix for consistency)
    output_transcript_json_path = output_dir / f"{pii_safe_file_prefix}_transcription.json"
    
    try:
        with open(output_transcript_json_path, 'w', encoding='utf-8') as f:
            # The structure here should match what downstream processes expect.
            # This typically is a list of segments.
            json.dump(transcribed_segments_data, f, indent=2, ensure_ascii=False)
        logger.info(f"[{pii_safe_file_prefix}] Full transcript with soundbite paths saved to: {output_transcript_json_path}")
        
        # Collect information about soundbite directories for the return value
        soundbite_dirs = list(set([str(Path(seg['soundbite_audio_path']).parent) for seg in transcribed_segments_data if 'soundbite_audio_path' in seg]))
        
        return {
            "transcript_json_path": str(output_transcript_json_path.resolve()),
            "soundbite_speaker_dirs": soundbite_dirs, # List of speaker directories containing .wav/.txt
            "message": f"Transcription complete. {len(transcribed_segments_data)} segments processed."
        }

    except IOError as e_io:
        logger.error(f"[{pii_safe_file_prefix}] Failed to write transcript JSON file '{output_transcript_json_path}': {e_io}", exc_info=True)
        return {"error": f"Failed to write transcript JSON: {e_io}", "transcript_json_path": None, "soundbite_dirs_info": None}
    except Exception as e_final:
        logger.error(f"[{pii_safe_file_prefix}] An unexpected error occurred during final output generation: {e_final}", exc_info=True)
        return {"error": f"Unexpected error in final output: {e_final}", "transcript_json_path": None, "soundbite_dirs_info": None}


# --- Main block for testing ---
if __name__ == '__main__':
    # This is a basic test setup.
    # You'll need:
    # 1. A sample vocal stem WAV file.
    # 2. A corresponding RTTM diarization file for that vocal stem.
    # 3. Adjust paths below accordingly.
    # 4. Ensure Whisper and Pyannote models can be downloaded or are cached.
    # 5. Set HF_TOKEN environment variable if your Pyannote models require it.
    
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] - %(name)s.%(funcName)s - %(message)s')
    logger.info("Starting transcription_module.py test run...")

    # --- Test Configuration ---
    # Create dummy vocal stem and RTTM for testing if they don't exist
    # For a real test, replace these with actual file paths.
    test_output_base_dir = Path("workspace/transcription_module_test_outputs")
    test_output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Dummy PII-safe prefix for testing
    test_pii_prefix = "call_20240101-120000_test"

    # Paths to your test files
    # IMPORTANT: Create these files or point to existing ones for the test to run.
    # For example, a short vocal recording and a simple RTTM file.
    
    # Example: Create a dummy 1-second silent WAV file for vocal_stem_path
    dummy_vocal_stem_path = test_output_base_dir / f"{test_pii_prefix}_vocals_normalized.wav"
    if not dummy_vocal_stem_path.exists():
        try:
            from pydub import AudioSegment
            silence = AudioSegment.silent(duration=1000, frame_rate=16000) # 1 sec, 16kHz
            silence.export(dummy_vocal_stem_path, format="wav")
            logger.info(f"Created dummy vocal stem: {dummy_vocal_stem_path}")
        except Exception as e:
            logger.error(f"Could not create dummy vocal stem: {e}. Please create it manually or provide a real one.")
            exit()
            
    # Example: Create a dummy RTTM file for diarization_file_path
    # This RTTM defines one speaker (S0) speaking for the first second.
    dummy_rttm_path = test_output_base_dir / f"{test_pii_prefix}_vocals_normalized_diarization.rttm"
    if not dummy_rttm_path.exists():
        rttm_content = f"SPEAKER {Path(dummy_vocal_stem_path).stem} 1 0.000 1.000 <NA> <NA> SPEAKER_00 <NA> <NA>\\n"
        # The file ID in RTTM (second field) should ideally match the stem of the audio file it refers to.
        # Here, Path(dummy_vocal_stem_path).stem would be 'call_20240101-120000_test_vocals_normalized'
        with open(dummy_rttm_path, 'w') as f_rttm:
            f_rttm.write(rttm_content)
        logger.info(f"Created dummy RTTM: {dummy_rttm_path}")

    # Test output directory for this run
    current_test_output_dir = test_output_base_dir / "run_01_transcription_stage_output"
    current_test_output_dir.mkdir(parents=True, exist_ok=True)
    
    test_config = {
        "whisper_model_name": "tiny.en",  # Use a small model for faster testing
        "language_for_transcription": "en",
        "min_segment_duration_ms_slicing": 100 # Low for dummy audio
    }

    logger.info(f"--- Test Parameters ---")
    logger.info(f"Vocal Stem: {dummy_vocal_stem_path}")
    logger.info(f"RTTM File: {dummy_rttm_path}")
    logger.info(f"Output Dir: {current_test_output_dir}")
    logger.info(f"PII Prefix: {test_pii_prefix}")
    logger.info(f"Config: {test_config}")
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
            logger.info(f"Transcription main JSON created at: {result['transcript_json_path']}")
            logger.info(f"Soundbite speaker directories: {result.get('soundbite_speaker_dirs')}")
            logger.info(f"Message: {result.get('message')}")
            
            # You can inspect the contents of current_test_output_dir to verify soundbites
            logger.info(f"Please inspect the contents of: {current_test_output_dir}")
            logger.info(f"And the main JSON: {result['transcript_json_path']}")
        else:
            logger.error(f"Test run_transcription failed or returned unexpected result: {result}")

    logger.info("Transcription_module.py test run finished.") 