# audio_analyzer_phase2.py
import os
import logging
import subprocess
from pathlib import Path
from datetime import datetime
import shutil
import json

# Attempt to import necessary components; these will need to be available in the environment
try:
    from audiomentations import Compose, LoudnessNormalization
    import soundfile as sf
    import numpy as np
except ImportError:
    logging.warning("audiomentations, soundfile, or numpy not found. Normalization will be a placeholder.")
    Compose, LoudnessNormalization, sf, np = None, None, None, None

try:
    from v3_8.ClapAnnotator.clap_annotation.annotator import CLAPAnnotatorWrapper
    # Assuming clap_settings_v3_8 can be accessed or its values are hardcoded/configured
    # from v3_8.ClapAnnotator.config import settings as clap_config_v3_8 
except ImportError:
    logging.warning("CLAPAnnotatorWrapper or its config not found. CLAP annotation will be a placeholder.")
    CLAPAnnotatorWrapper = None
    # clap_config_v3_8 = None

try:
    import whisper
except ImportError:
    logging.warning("OpenAI Whisper not found. Transcription will be a placeholder.")
    whisper = None

# --- Constants ---
DEFAULT_VOCAL_STEM_TARGET_LUFS_PH2 = 0.0
DEFAULT_INSTRUMENTAL_STEM_TARGET_LUFS_PH2 = -14.0
CLAP_EXPECTED_SR_PH2 = 48000 
# CLAP_MODEL_NAME_PH2 = clap_config_v3_8.CLAP_MODEL_NAME if clap_config_v3_8 else "laion/clap-htsat-fused"
# CLAP_CHUNK_DURATION_S_PH2 = clap_config_v3_8.CLAP_CHUNK_DURATION_S if clap_config_v3_8 else 3
# CLAP_CONFIDENCE_PH2 = clap_config_v3_8.DEFAULT_CLAP_CONFIDENCE_THRESHOLD if clap_config_v3_8 else 0.55
CLAP_MODEL_NAME_PH2 = "laion/clap-htsat-fused" # Fallback if clap_config not loaded
CLAP_CHUNK_DURATION_S_PH2 = 3
CLAP_DEFAULT_CONFIDENCE_PH2 = 0.55


DEFAULT_WHISPER_MODEL_PH2 = "large-v3"
# Consistent with audiotoolkit_phase1.py and ClapAnnotator default
AUDIO_SEPARATOR_MODEL_NAME_PH2 = "mel_band_roformer_vocals_fv4_gabox.ckpt" 
DEFAULT_AUDIO_SAMPLE_RATE_PH2 = 44100 # Standard intermediate sample rate

# --- Utility Functions ---
def setup_logging_phase2(log_level_str='INFO'):
    """Configures logging for the Phase 2 analyzer."""
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s [%(levelname)s] - %(module)s.%(funcName)s - %(message)s')
    logging.getLogger("httpx").setLevel(logging.WARNING) # To silence noisy httpx logs from Hugging Face a bit


def run_ffmpeg_command_phase2(command, description):
    """Run an FFmpeg command with proper logging for Phase 2."""
    logging.info(f"Running FFmpeg for: {description} | CMD: {' '.join(command)}")
    try:
        process = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logging.debug(f"FFmpeg stdout for {description}:\n{process.stdout}")
        logging.debug(f"FFmpeg stderr for {description}:\n{process.stderr}")
        logging.info(f"{description} completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"{description} failed with error: {e}")
        logging.error(f"FFmpeg stdout (error):\n{e.stdout}")
        logging.error(f"FFmpeg stderr (error):\n{e.stderr}")
        return False
    except Exception as ex:
        logging.error(f"Unexpected error during {description}: {ex}")
        return False

def resample_audio_ffmpeg_phase2(input_path: Path, output_path: Path, target_sr: int, mono: bool = True):
    """Uses ffmpeg to resample and optionally convert to mono."""
    logging.info(f"Resampling {input_path} to {target_sr}Hz (mono={mono}) -> {output_path}")
    command = ["ffmpeg", "-y", "-i", str(input_path)]
    audio_filters = [f"aresample={target_sr}"]
    if mono:
        command.extend(["-ac", "1"]) # Set 1 audio channel for mono
    
    # Apply filter complex for resampling
    command.extend(["-af", ",".join(audio_filters)])
    command.append(str(output_path))
    
    if run_ffmpeg_command_phase2(command, f"Resampling {input_path.name} to {target_sr}Hz"):
        return output_path
    return None

# --- Core Processing Functions ---

def separate_stems_phase2(input_audio: Path, temp_dir: Path, separator_model_name: str):
    """
    Separates audio into vocals and instrumental stems using audio-separator CLI.
    Outputs raw stems to a subdirectory in temp_dir.
    Returns dict: {'vocals': path_to_vocals, 'instrumental': path_to_instrumental} or None on failure.
    """
    logging.info(f"Separating stems for: {input_audio} using model: {separator_model_name}")
    stems_output_dir = temp_dir / "00_separated_stems_raw"
    stems_output_dir.mkdir(parents=True, exist_ok=True)

    # audio-separator expects input file to be in the output_dir or specified with full path.
    # To get predictable output names for renaming, we copy input to a temp name inside stems_output_dir.
    temp_input_basename = "input_for_separator"
    temp_input_ext = input_audio.suffix
    temp_separator_input_path = stems_output_dir / f"{temp_input_basename}{temp_input_ext}"
    
    try:
        shutil.copy2(input_audio, temp_separator_input_path)
    except Exception as e:
        logging.error(f"Failed to copy {input_audio} to {temp_separator_input_path}: {e}")
        return None

    cmd = [
        "audio-separator", # Assuming audio-separator is in PATH
        str(temp_separator_input_path), # Process the copied temp file
        "-m", separator_model_name,
        "--output_dir", str(stems_output_dir), # Output next to the temp input
        "--output_format", "wav",
        "--log_level", "DEBUG" # Or INFO
        # Add other params like --mdx_segment_size=256 if needed for specific models
    ]
    
    logging.info(f"Running audio-separator command: {' '.join(cmd)}")
    
    final_raw_vocals_path = stems_output_dir / "vocals_raw.wav"
    final_raw_instrumental_path = stems_output_dir / "instrumental_raw.wav"
    
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.debug(f"audio-separator stdout:\n{process.stdout}")
        # audio-separator stderr can be very verbose with DEBUG, so maybe only log on error or if needed.
        # logging.debug(f"audio-separator stderr:\n{process.stderr}") 
        logging.info(f"audio-separator completed for {temp_separator_input_path}.")

        # Default output name pattern from audio-separator: <input_base>_(StemName)_<model_name_without_ckpt_ext>.wav
        model_name_stem_for_output = Path(separator_model_name).stem # e.g., "mel_band_roformer_vocals_fv4_gabox"
        
        # Example: input_for_separator_(Vocals)_mel_band_roformer_vocals_fv4_gabox.wav
        generated_vocals_filename = f"{temp_input_basename}_(Vocals)_{model_name_stem_for_output}.wav"
        generated_instrumental_filename = f"{temp_input_basename}_(Instrumental)_{model_name_stem_for_output}.wav"

        temp_vocals_path = stems_output_dir / generated_vocals_filename
        temp_instrumental_path = stems_output_dir / generated_instrumental_filename
        
        stem_paths_found = {}
        if temp_vocals_path.exists():
            os.rename(str(temp_vocals_path), str(final_raw_vocals_path))
            stem_paths_found['vocals'] = final_raw_vocals_path
            logging.info(f"Renamed separated vocals to: {final_raw_vocals_path}")
        else:
            logging.warning(f"Separated vocals stem not found at expected path: {temp_vocals_path}")
            # Attempt to find any vocals file if naming is different
            found_vocals = list(stems_output_dir.glob(f"{temp_input_basename}_*Vocals*.wav"))
            if found_vocals:
                os.rename(str(found_vocals[0]), str(final_raw_vocals_path))
                stem_paths_found['vocals'] = final_raw_vocals_path
                logging.info(f"Found and renamed vocals: {final_raw_vocals_path}")
            else:
                 logging.error("No vocal stem produced.")
                 return None # Critical if no vocals

        if temp_instrumental_path.exists():
            os.rename(str(temp_instrumental_path), str(final_raw_instrumental_path))
            stem_paths_found['instrumental'] = final_raw_instrumental_path
            logging.info(f"Renamed separated instrumental to: {final_raw_instrumental_path}")
        else:
            logging.warning(f"Separated instrumental stem not found at expected path: {temp_instrumental_path}")
            found_instrumentals = list(stems_output_dir.glob(f"{temp_input_basename}_*Instrumental*.wav"))
            if found_instrumentals:
                os.rename(str(found_instrumentals[0]), str(final_raw_instrumental_path))
                stem_paths_found['instrumental'] = final_raw_instrumental_path
                logging.info(f"Found and renamed instrumental: {final_raw_instrumental_path}")
            else:
                logging.info("No instrumental stem produced (this might be okay for some inputs).")
        
        if 'vocals' not in stem_paths_found: # Vocals are essential
            logging.error("Could not find or rename vocal stem. Aborting separation.")
            return None
            
        return stem_paths_found

    except subprocess.CalledProcessError as e:
        logging.error(f"audio-separator failed for {input_audio}.")
        logging.error(f"Stdout: {e.stdout}")
        logging.error(f"Stderr: {e.stderr}") # Log stderr on error
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during stem separation: {e}")
        return None
    finally:
        # Clean up the temporary input file we created for the separator
        if temp_separator_input_path.exists():
            try:
                os.remove(temp_separator_input_path)
            except OSError as e_rm:
                logging.warning(f"Could not remove temporary separator input {temp_separator_input_path}: {e_rm}")
        # Do not clean up stems_output_dir here as it contains the final raw stems


def normalize_stem_audiomentations_phase2(input_path: Path, output_path: Path, target_lufs: float, target_sr: int = DEFAULT_AUDIO_SAMPLE_RATE_PH2):
    """
    Normalizes a stem to target_lufs using audiomentations.LoudnessNormalization.
    Ensures output is at target_sr and mono.
    """
    if not all([Compose, LoudnessNormalization, sf, np]):
        logging.warning("Audiomentations components not available. Skipping normalization. Copying file instead.")
        if input_path != output_path: shutil.copy2(input_path, output_path)
        # Further ensure it's at target_sr if possible, even if not normalized.
        # This might require a simple ffmpeg resample if audiomentations is skipped.
        return output_path

    logging.info(f"Normalizing (audiomentations) {input_path.name} to {target_lufs} LUFS, target SR {target_sr}Hz -> {output_path.name}")
    try:
        samples, original_sr = sf.read(input_path, dtype='float32', always_2d=False)
        
        # Ensure mono
        if samples.ndim > 1 and samples.shape[-1] > 1: # Check last dimension for channels
             samples = np.mean(samples, axis=-1)
             logging.debug(f"Mixed {input_path.name} to mono for normalization.")

        # Resample if necessary before normalization (audiomentations LNorm doesn't change SR)
        if original_sr != target_sr:
            logging.debug(f"Resampling {input_path.name} from {original_sr}Hz to {target_sr}Hz before normalization.")
            # This requires a temp file or in-memory resampling if librosa is available
            # For simplicity, let's assume ffmpeg utility for this before audiomentations if needed,
            # or ensure input to this function is already at target_sr.
            # Current approach: Resample with ffmpeg *before* calling this normalize function if SR change is needed.
            # However, if called directly, this function should handle it.
            # Let's use ffmpeg for resampling here for robustness if sr mismatch.
            temp_resampled_path = output_path.with_suffix(f".temp_resample{output_path.suffix}")
            resample_ok = resample_audio_ffmpeg_phase2(input_path, temp_resampled_path, target_sr, mono=True)
            if not resample_ok:
                logging.error(f"Failed to resample {input_path} to {target_sr}Hz. Skipping normalization.")
                if input_path != output_path: shutil.copy2(input_path, output_path)
                return output_path
            samples, original_sr = sf.read(temp_resampled_path, dtype='float32') # sr is now target_sr
            os.remove(temp_resampled_path)


        augment = Compose([
            LoudnessNormalization(min_lufs_in_db=float(target_lufs), max_lufs_in_db=float(target_lufs), p=1.0)
        ])
        processed_samples = augment(samples=samples, sample_rate=original_sr) # original_sr is now target_sr
        
        sf.write(output_path, processed_samples, original_sr) # Write with the (potentially new) target_sr
        logging.info(f"Successfully normalized and wrote to {output_path} at {original_sr}Hz.")
        return output_path
    except Exception as e:
        logging.error(f"Error during audiomentations LoudnessNormalization for {input_path}: {e}")
        if input_path != output_path: shutil.copy2(input_path, output_path) # Fallback
        return output_path # Return original or copied path


def analyze_audio_phase2(input_audio_path_str: str, output_base_dir_str: str, config: dict = None):
    if config is None:
        config = {}
    
    input_audio_path = Path(input_audio_path_str)
    output_base_dir = Path(output_base_dir_str)

    run_dt = datetime.now()
    input_filename_stem = input_audio_path.stem
    sanitized_input_stem = "".join(c if c.isalnum() else "_" for c in input_filename_stem)
    run_id = f"{sanitized_input_stem}_{run_dt.strftime('%Y%m%d_%H%M%S')}"
    
    current_run_output_dir = output_base_dir / run_id
    current_run_temp_dir = current_run_output_dir / "temp_processing"
    current_run_output_dir.mkdir(parents=True, exist_ok=True)
    current_run_temp_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Starting Phase 2 analysis for {input_audio_path}. Run ID: {run_id}")
    logging.info(f"Output directory: {current_run_output_dir}")

    final_results_summary = {"run_id": run_id, "input_file": str(input_audio_path), "clap_events": {}, "transcription": {}}

    # --- 1. Stem Separation ---
    separator_model = config.get('separator_model_ph2', AUDIO_SEPARATOR_MODEL_NAME_PH2)
    logging.info(f"Step 1: Separating stems using {separator_model}...")
    separated_stem_paths = separate_stems_phase2(input_audio_path, current_run_temp_dir, separator_model)
    
    if not separated_stem_paths or not separated_stem_paths.get('vocals'):
        logging.error(f"Could not separate vocals for {input_audio_path}. Aborting.")
        return None
    
    vocals_raw_path = separated_stem_paths['vocals']
    instrumental_raw_path = separated_stem_paths.get('instrumental') # Might be None
    logging.info(f"Vocals separated to: {vocals_raw_path}")
    if instrumental_raw_path: logging.info(f"Instrumental separated to: {instrumental_raw_path}")


    # --- 2. Stem Normalization (Audiomentations) ---
    # Ensure output is DEFAULT_AUDIO_SAMPLE_RATE_PH2 (e.g., 44100 Hz) for consistency before CLAP's specific resampling.
    target_norm_sr = config.get('normalized_stem_sr_ph2', DEFAULT_AUDIO_SAMPLE_RATE_PH2)
    logging.info(f"Step 2: Normalizing stems to target SR {target_norm_sr}Hz...")

    vocals_norm_lufs = config.get('vocal_stem_target_lufs_ph2', DEFAULT_VOCAL_STEM_TARGET_LUFS_PH2)
    vocals_norm_path = current_run_temp_dir / f"01_vocals_norm_{target_norm_sr}Hz.wav"
    normalize_stem_audiomentations_phase2(vocals_raw_path, vocals_norm_path, vocals_norm_lufs, target_sr=target_norm_sr)
    shutil.copy2(vocals_norm_path, current_run_output_dir / vocals_norm_path.name) # Save a copy to main output

    instrumental_norm_path = None
    if instrumental_raw_path and instrumental_raw_path.exists():
        instrumental_norm_lufs = config.get('instrumental_stem_target_lufs_ph2', DEFAULT_INSTRUMENTAL_STEM_TARGET_LUFS_PH2)
        instrumental_norm_path = current_run_temp_dir / f"01_instrumental_norm_{target_norm_sr}Hz.wav"
        normalize_stem_audiomentations_phase2(instrumental_raw_path, instrumental_norm_path, instrumental_norm_lufs, target_sr=target_norm_sr)
        shutil.copy2(instrumental_norm_path, current_run_output_dir / instrumental_norm_path.name) # Save a copy

    # --- 3. Prepare Stems for CLAP & Annotate ---
    clap_expected_sr_actual = config.get('clap_expected_sr_ph2', CLAP_EXPECTED_SR_PH2)
    clap_model_name_actual = config.get('clap_model_name_ph2', CLAP_MODEL_NAME_PH2)
    clap_chunk_duration_actual = config.get('clap_chunk_duration_s_ph2', CLAP_CHUNK_DURATION_S_PH2)
    clap_confidence_actual = config.get('clap_confidence_ph2', CLAP_DEFAULT_CONFIDENCE_PH2)
    clap_prompts = config.get('clap_prompts_ph2', ["speech", "music", "silence", "applause", "laughter", "noise", "telephone"])
    
    logging.info(f"Step 3: Preparing stems for CLAP (target SR: {clap_expected_sr_actual}Hz) & Annotating...")

    clap_annotator_instance = None
    if CLAPAnnotatorWrapper:
        try:
            # This assumes CLAPAnnotatorWrapper can find its models etc.
            # May need to pass model dir or ensure HF cache is set up.
            clap_annotator_instance = CLAPAnnotatorWrapper(
                model_name=clap_model_name_actual,
                chunk_duration_s=clap_chunk_duration_actual,
                expected_sr=clap_expected_sr_actual
            )
            logging.info(f"CLAPAnnotatorWrapper initialized with model {clap_model_name_actual}.")
        except Exception as e_clap_init:
            logging.error(f"Failed to initialize CLAPAnnotatorWrapper: {e_clap_init}. CLAP will be skipped.")
            clap_annotator_instance = None
    else:
        logging.warning("CLAPAnnotatorWrapper not available. CLAP annotation will be skipped.")

    # Process Vocals with CLAP
    if vocals_norm_path.exists():
        vocals_clap_input_path = current_run_temp_dir / f"02_vocals_for_clap_{clap_expected_sr_actual}Hz.wav"
        resample_audio_ffmpeg_phase2(vocals_norm_path, vocals_clap_input_path, clap_expected_sr_actual, mono=True)
        
        if clap_annotator_instance and vocals_clap_input_path.exists():
            logging.info(f"Running CLAP on vocal stem: {vocals_clap_input_path}")
            try:
                vocals_clap_data = clap_annotator_instance.annotate(vocals_clap_input_path, clap_prompts, clap_confidence_actual)
                final_results_summary["clap_events"]["vocals"] = vocals_clap_data 
                with open(current_run_output_dir / "vocals_clap_events.json", "w") as f_json:
                   json.dump(vocals_clap_data, f_json, indent=2)
                logging.info(f"Vocal CLAP events saved to vocals_clap_events.json. Detections: {len(vocals_clap_data.get('detections',[]))}")
            except Exception as e_clap_voc:
                logging.error(f"Error during vocal CLAP annotation: {e_clap_voc}")
                final_results_summary["clap_events"]["vocals"] = {"error": str(e_clap_voc)}
        elif not clap_annotator_instance:
             logging.warning("Vocal CLAP annotation skipped as annotator failed to initialize.")
        elif not vocals_clap_input_path.exists():
            logging.warning(f"Resampled vocal file for CLAP not found: {vocals_clap_input_path}")


    # Process Instrumentals with CLAP
    if instrumental_norm_path and instrumental_norm_path.exists():
        instrumental_clap_input_path = current_run_temp_dir / f"02_instrumental_for_clap_{clap_expected_sr_actual}Hz.wav"
        resample_audio_ffmpeg_phase2(instrumental_norm_path, instrumental_clap_input_path, clap_expected_sr_actual, mono=True)

        if clap_annotator_instance and instrumental_clap_input_path.exists():
            logging.info(f"Running CLAP on instrumental stem: {instrumental_clap_input_path}")
            try:
                instrumental_clap_data = clap_annotator_instance.annotate(instrumental_clap_input_path, clap_prompts, clap_confidence_actual)
                final_results_summary["clap_events"]["instrumental"] = instrumental_clap_data
                with open(current_run_output_dir / "instrumental_clap_events.json", "w") as f_json:
                   json.dump(instrumental_clap_data, f_json, indent=2)
                logging.info(f"Instrumental CLAP events saved to instrumental_clap_events.json. Detections: {len(instrumental_clap_data.get('detections',[]))}")
            except Exception as e_clap_inst:
                logging.error(f"Error during instrumental CLAP annotation: {e_clap_inst}")
                final_results_summary["clap_events"]["instrumental"] = {"error": str(e_clap_inst)}
        elif not clap_annotator_instance:
            logging.warning("Instrumental CLAP annotation skipped as annotator failed to initialize.")
        elif not instrumental_clap_input_path.exists():
             logging.warning(f"Resampled instrumental file for CLAP not found: {instrumental_clap_input_path}")


    # --- 4. Transcription using Whisper ---
    whisper_model_name = config.get('whisper_model_ph2', DEFAULT_WHISPER_MODEL_PH2)
    logging.info(f"Step 4: Transcribing vocal stem using Whisper model: {whisper_model_name}...")
    
    if vocals_norm_path.exists(): # Transcribe the normalized vocals (e.g., at 44.1kHz or target_norm_sr)
        if whisper:
            try:
                logging.info(f"Loading Whisper model: {whisper_model_name}")
                # Consider making model download location configurable or using system cache
                whisper_model_instance = whisper.load_model(whisper_model_name)
                logging.info(f"Transcribing {vocals_norm_path}...")
                transcription_result = whisper_model_instance.transcribe(str(vocals_norm_path), word_timestamps=True)
                
                final_results_summary["transcription"] = transcription_result
                with open(current_run_output_dir / "transcript.json", "w", encoding='utf-8') as f_json:
                   json.dump(transcription_result, f_json, indent=2, ensure_ascii=False)
                with open(current_run_output_dir / "transcript.txt", "w", encoding='utf-8') as f_text:
                   f_text.write(transcription_result.get("text", ""))
                # Basic SRT generation
                with open(current_run_output_dir / "transcript.srt", "w", encoding='utf-8') as f_srt:
                    for i, segment in enumerate(transcription_result.get("segments", [])):
                        start_time = segment['start']
                        end_time = segment['end']
                        text = segment['text'].strip()
                        f_srt.write(f"{i+1}\n")
                        f_srt.write(f"{str(datetime.utcfromtimestamp(start_time).strftime('%H:%M:%S,%f')[:-3])} --> {str(datetime.utcfromtimestamp(end_time).strftime('%H:%M:%S,%f')[:-3])}\n")
                        f_srt.write(f"{text}\n\n")

                logging.info(f"Transcription saved (json, txt, srt).")
            except Exception as e_whisper:
                logging.error(f"Error during Whisper transcription: {e_whisper}")
                final_results_summary["transcription"] = {"error": str(e_whisper)}
        else:
            logging.warning("OpenAI Whisper not available. Transcription skipped.")
            final_results_summary["transcription"] = {"text": "Whisper transcription skipped (library not found).", "segments": [], "language": "en"}
    else:
        logging.warning("Normalized vocal stem not found, cannot transcribe.")
        final_results_summary["transcription"] = {"text": "Transcription skipped (no vocal stem).", "segments": [], "language": "en"}


    # --- 5. Finalize and Save Overall Results ---
    with open(current_run_output_dir / "analysis_summary.json", "w", encoding='utf-8') as f_summary:
        json.dump(final_results_summary, f_summary, indent=2, ensure_ascii=False)
        
    logging.info(f"Phase 2 analysis complete for {run_id}. Results in {current_run_output_dir}")
    
    # Optional: cleanup temp_dir
    cleanup_temp = config.get('cleanup_temp_dirs_ph2', True)
    if cleanup_temp:
        logging.info(f"Cleaning up temporary directory: {current_run_temp_dir}")
        try:
            shutil.rmtree(current_run_temp_dir)
        except Exception as e_clean:
            logging.warning(f"Could not cleanup temp directory {current_run_temp_dir}: {e_clean}")

    return str(current_run_output_dir)


if __name__ == "__main__":
    # Example Usage
    setup_logging_phase2(log_level_str="INFO") 
    
    # Create a dummy stereo WAV file for testing if one doesn't exist
    # Note: audio-separator works best with actual vocal/instrumental content.
    # A simple sine wave might not separate well or at all.
    # Using a real, short audio clip is recommended for testing separation.
    
    example_input_filename = "example_phase2_input.wav"
    test_input_file = Path(example_input_filename)

    # This basic test requires a real audio file named example_phase2_input.wav in the same directory.
    # For a quick test without real audio, many steps will be skipped or use placeholders.
    if not test_input_file.exists():
        logging.warning(f"Test input file '{example_input_filename}' not found in the current directory.")
        logging.warning("Please create a suitable WAV file for testing (e.g., a short song clip).")
        # Fallback: create a very simple dummy file if soundfile is available
        if sf and np:
            try:
                samplerate = 44100
                duration = 3 # seconds
                frequency = 440 # Hz (A4 note)
                t = np.linspace(0, duration, int(samplerate * duration), False)
                note = 0.5 * np.sin(2 * np.pi * frequency * t) # Mono
                # Add a bit of another frequency to simulate "instrumental"
                note += 0.3 * np.sin(2 * np.pi * (frequency * 1.5) * t)
                sf.write(test_input_file, note, samplerate)
                logging.info(f"Created a simple dummy test file: {test_input_file} (separation may not be meaningful).")
            except Exception as e_create_dummy:
                 logging.error(f"Could not create dummy test file: {e_create_dummy}. Please provide a {example_input_filename}.")
                 exit()
        else:
            logging.error("Soundfile/numpy not available to create a dummy file. Exiting.")
            exit()


    if test_input_file.exists():
        output_directory_main = "phase2_analysis_outputs"
        
        # Config for testing
        # Note: Actual model loading for CLAP/Whisper can be slow/resource-intensive.
        # The code has placeholders if libraries are missing.
        test_run_config = {
            "separator_model_ph2": AUDIO_SEPARATOR_MODEL_NAME_PH2,
            "vocal_stem_target_lufs_ph2": -10.0, 
            "instrumental_stem_target_lufs_ph2": -18.0, 
            "normalized_stem_sr_ph2": 44100, # SR after normalization, before CLAP resampling
            "clap_expected_sr_ph2": 48000,
            "clap_model_name_ph2": "laion/clap-htsat-fused",
            "clap_chunk_duration_s_ph2": 3,
            "clap_prompts_ph2": ["speech", "music", "dog barking", "car horn", "silence"],
            "clap_confidence_ph2": 0.3, # Lower for testing to get more hits potentially
            "whisper_model_ph2": "tiny", # Use a small model for faster testing
            "cleanup_temp_dirs_ph2": False # Keep temp files for inspection
        }
        
        logging.info(f"Running Phase 2 analysis with test config on: {test_input_file}")
        analyze_audio_phase2(str(test_input_file), output_directory_main, config=test_run_config)
    else:
        logging.error(f"Test input file '{example_input_filename}' still not found. Cannot run example.") 