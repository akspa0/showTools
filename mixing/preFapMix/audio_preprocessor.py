# audio_preprocessor.py
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

# --- Constants for audio_preprocessor ---
DEFAULT_VOCAL_TARGET_LUFS_CORE = 0.0
DEFAULT_INSTRUMENTAL_TARGET_LUFS_CORE = -14.0
DEFAULT_TARGET_SR_CORE = 44100 # Target sample rate for normalized stems
DEFAULT_SEPARATOR_MODEL_CORE = "mel_band_roformer_vocals_fv4_gabox.ckpt"

# --- Utility Functions ---
def setup_logging_core(log_level_str='INFO', module_name='audio_preprocessor'):
    """Configures logging for the audio_preprocessor module."""
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    # Use a distinct logger name for this module if desired, or use root logger
    logger = logging.getLogger(module_name)
    logger.setLevel(log_level)
    if not logger.hasHandlers(): # Avoid adding multiple handlers if called multiple times
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(name)s.%(funcName)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    # Silence overly verbose library logs if necessary
    logging.getLogger("httpx").setLevel(logging.WARNING) 
    return logger # Return the logger instance

log = setup_logging_core() # Initialize logger for this module

def run_ffmpeg_command_core(command, description):
    """Run an FFmpeg command with proper logging."""
    log.info(f"Running FFmpeg for: {description} | CMD: {' '.join(command)}")
    try:
        process = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        log.debug(f"""FFmpeg stdout for {description}:
{process.stdout}""")
        log.debug(f"""FFmpeg stderr for {description}:
{process.stderr}""")
        log.info(f"{description} completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"{description} failed with error: {e}")
        log.error(f"""FFmpeg stdout (error):
{e.stdout}""")
        log.error(f"""FFmpeg stderr (error):
{e.stderr}""")
        return False
    except Exception as ex:
        log.error(f"Unexpected error during {description}: {ex}")
        return False

def resample_audio_ffmpeg_core(input_path: Path, output_path: Path, target_sr: int, mono: bool = True):
    """Uses ffmpeg to resample and optionally convert to mono."""
    log.info(f"Resampling {input_path} to {target_sr}Hz (mono={mono}) -> {output_path}")
    command = ["ffmpeg", "-y", "-i", str(input_path)]
    audio_filters = [f"aresample={target_sr}"]
    if mono:
        command.extend(["-ac", "1"])
    
    command.extend(["-af", ",".join(audio_filters)])
    command.append(str(output_path))
    
    if run_ffmpeg_command_core(command, f"Resampling {input_path.name} to {target_sr}Hz"):
        return output_path
    return None

# --- Core Processing Functions ---

def separate_stems_core(input_audio: Path, base_temp_dir: Path, separator_model_name: str, pii_safe_file_prefix: str, config: dict = None):
    """
    Separates audio into vocals and instrumental stems using audio-separator CLI.
    Outputs raw stems to a subdirectory in base_temp_dir.
    Accepts pii_safe_file_prefix for naming temporary and output files.
    Returns dict: {'vocals': path_to_vocals, 'instrumental': path_to_instrumental} or None on failure.
    """
    log.info(f"Separating stems for: {input_audio} (using prefix: {pii_safe_file_prefix}) using model: {separator_model_name}")
    # Create a unique subdir for this separation task's raw outputs, using PII-safe prefix
    stems_output_dir = base_temp_dir / f"separated_raw_{pii_safe_file_prefix}"
    stems_output_dir.mkdir(parents=True, exist_ok=True)

    # Using a generic temp input name to avoid issues with special chars in original name for audio-separator
    temp_input_basename = f"{pii_safe_file_prefix}_temp_sep_input" 
    temp_input_ext = input_audio.suffix
    temp_separator_input_path = stems_output_dir / f"{temp_input_basename}{temp_input_ext}"
    
    try:
        shutil.copy2(input_audio, temp_separator_input_path)
    except Exception as e:
        log.error(f"Failed to copy {input_audio} to {temp_separator_input_path}: {e}")
        return None

    cmd = [
        "audio-separator",
        str(temp_separator_input_path),
        "-m", separator_model_name,
        "--output_dir", str(stems_output_dir),
        "--output_format", "wav",
        "--log_level", "DEBUG"
    ]
    
    log.info(f"Running audio-separator command: {' '.join(cmd)}")
    
    # Predictable final names within this function's scope, using PII-safe prefix
    final_raw_vocals_path = stems_output_dir / f"{pii_safe_file_prefix}_vocals_raw.wav"
    final_raw_instrumental_path = stems_output_dir / f"{pii_safe_file_prefix}_instrumental_raw.wav"
    
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        log.debug(f"""audio-separator stdout:
{process.stdout}""")
        log.info(f"audio-separator completed for {temp_separator_input_path}.")

        model_name_stem_for_output = Path(separator_model_name).stem
        # Output from audio-separator uses the *actual input filename* it was given
        generated_vocals_filename = f"{temp_input_basename}_(Vocals)_{model_name_stem_for_output}.wav"
        generated_instrumental_filename = f"{temp_input_basename}_(Instrumental)_{model_name_stem_for_output}.wav"

        temp_vocals_path = stems_output_dir / generated_vocals_filename
        temp_instrumental_path = stems_output_dir / generated_instrumental_filename
        
        stem_paths_found = {}
        if temp_vocals_path.exists():
            os.rename(str(temp_vocals_path), str(final_raw_vocals_path))
            stem_paths_found['vocals'] = final_raw_vocals_path
            log.info(f"Renamed separated vocals to: {final_raw_vocals_path}")
        else:
            found_vocals = list(stems_output_dir.glob(f"{temp_input_basename}_*Vocals*.wav"))
            if found_vocals:
                os.rename(str(found_vocals[0]), str(final_raw_vocals_path))
                stem_paths_found['vocals'] = final_raw_vocals_path
                log.info(f"Found and renamed vocals: {final_raw_vocals_path}")
            else:
                 log.error(f"No vocal stem produced from {temp_vocals_path} or via glob. Creating dummy vocal stem for pipeline continuation.")
                 # Create a dummy VALID silent vocal file
                 try:
                     # Ensure np and sf are available (checked at top of file)
                     if np and sf:
                         silent_duration_seconds = 1
                         # Use DEFAULT_TARGET_SR_CORE as a fallback if target_sr not directly available here
                         # Or, ideally, pass target_sr into separate_stems_core if needed for dummy generation
                         sample_rate_for_dummy = config.get('target_sr_core', DEFAULT_TARGET_SR_CORE) if isinstance(config, dict) else DEFAULT_TARGET_SR_CORE

                         num_channels = 1 # Mono for stems
                         num_frames = int(silent_duration_seconds * sample_rate_for_dummy)
                         silent_data = np.zeros((num_frames, num_channels) if num_channels > 1 else num_frames, dtype='float32')
                         sf.write(str(final_raw_vocals_path), silent_data, sample_rate_for_dummy)
                         log.info(f"Created dummy silent vocal stem: {final_raw_vocals_path} at {sample_rate_for_dummy}Hz")
                         stem_paths_found['vocals'] = final_raw_vocals_path
                     else:
                         log.error("numpy or soundfile not available, cannot create valid dummy silent WAV. Falling back to touch.")
                         Path(final_raw_vocals_path).touch()
                         log.info(f"Created (empty) dummy vocal stem (fallback): {final_raw_vocals_path}")
                         stem_paths_found['vocals'] = final_raw_vocals_path # Still provide path for structure
                 except Exception as e_dummy_vocal:
                     log.error(f"Failed to create dummy vocal stem {final_raw_vocals_path}: {e_dummy_vocal}")
                     return None # Critical failure if dummy cannot be made

        if temp_instrumental_path.exists():
            os.rename(str(temp_instrumental_path), str(final_raw_instrumental_path))
            stem_paths_found['instrumental'] = final_raw_instrumental_path
            log.info(f"Renamed separated instrumental to: {final_raw_instrumental_path}")
        else:
            found_instrumentals = list(stems_output_dir.glob(f"{temp_input_basename}_*Instrumental*.wav"))
            if found_instrumentals:
                os.rename(str(found_instrumentals[0]), str(final_raw_instrumental_path))
                stem_paths_found['instrumental'] = final_raw_instrumental_path
                log.info(f"Found and renamed instrumental: {final_raw_instrumental_path}")
            else:
                log.info("No instrumental stem produced. Creating dummy instrumental stem if vocals exist.")
                if 'vocals' in stem_paths_found: 
                    try:
                        if np and sf:
                            silent_duration_seconds = 1
                            sample_rate_for_dummy = config.get('target_sr_core', DEFAULT_TARGET_SR_CORE) if isinstance(config, dict) else DEFAULT_TARGET_SR_CORE
                            num_channels = 1 # Mono for stems
                            num_frames = int(silent_duration_seconds * sample_rate_for_dummy)
                            silent_data = np.zeros((num_frames, num_channels) if num_channels > 1 else num_frames, dtype='float32')
                            sf.write(str(final_raw_instrumental_path), silent_data, sample_rate_for_dummy)
                            log.info(f"Created dummy silent instrumental stem: {final_raw_instrumental_path} at {sample_rate_for_dummy}Hz")
                            stem_paths_found['instrumental'] = final_raw_instrumental_path
                        else:
                            log.error("numpy or soundfile not available, cannot create valid dummy silent WAV for instrumental. Falling back to touch.")
                            Path(final_raw_instrumental_path).touch()
                            log.info(f"Created (empty) dummy instrumental stem (fallback): {final_raw_instrumental_path}")
                            stem_paths_found['instrumental'] = final_raw_instrumental_path # Still provide path

                    except Exception as e_dummy_instr:
                        log.warning(f"Failed to create dummy instrumental stem {final_raw_instrumental_path}: {e_dummy_instr}")
        if 'vocals' not in stem_paths_found:
            log.error("Could not find or create vocal stem after separation attempt.")
            return None
            
        return stem_paths_found

    except subprocess.CalledProcessError as e:
        log.error(f"audio-separator failed for {input_audio}. CMD: {' '.join(cmd)}")
        log.error(f"""Stdout:
{e.stdout}""")
        log.error(f"""Stderr:
{e.stderr}""")
        return None
    except Exception as e:
        log.error(f"An unexpected error occurred during stem separation: {e}")
        return None
    finally:
        if temp_separator_input_path.exists():
            try:
                os.remove(temp_separator_input_path)
            except OSError as e_rm:
                log.warning(f"Could not remove temp separator input {temp_separator_input_path}: {e_rm}")

def normalize_stem_audiomentations_core(input_path: Path, output_path: Path, target_lufs: float, target_sr: int):
    """
    Normalizes a stem to target_lufs using audiomentations.LoudnessNormalization.
    Ensures output is at target_sr and mono.
    """
    if not all([Compose, LoudnessNormalization, sf, np]):
        log.warning("Audiomentations components not available. Skipping normalization. Copying file instead.")
        if input_path != output_path: shutil.copy2(input_path, output_path)
        # If copied, it might not be at target_sr. This should be handled by the caller or by an explicit resample.
        return output_path

    log.info(f"Normalizing (audiomentations) {input_path.name} to {target_lufs} LUFS, target SR {target_sr}Hz -> {output_path.name}")
    try:
        samples, original_sr = sf.read(input_path, dtype='float32', always_2d=False)
        
        current_samples_path = input_path
        current_sr_for_norm = original_sr

        # --- Explicit Resampling Step using ffmpeg --- 
        temp_resampled_path = None
        if original_sr != target_sr:
            log.debug(f"Resampling {input_path.name} from {original_sr}Hz to {target_sr}Hz before normalization.")
            # Create a temporary path for the resampled audio in the same directory as output_path
            temp_resampled_path = output_path.with_name(output_path.stem + "_temp_resampled.wav")
            if resample_audio_ffmpeg_core(input_path, temp_resampled_path, target_sr, mono=True):
                current_samples_path = temp_resampled_path
                current_sr_for_norm = target_sr
                log.debug(f"Successfully resampled to {temp_resampled_path}")
                # Samples will be re-read from this resampled file
            else:
                log.error(f"Failed to resample {input_path.name} to {target_sr}Hz. Attempting normalization with original SR: {original_sr}Hz.")
                # If resampling fails, continue with original samples and SR, ffmpeg peak limiter will handle final SR.
        
        # Read potentially resampled audio for audiomentations
        # If resampling happened, current_samples_path is temp_resampled_path, otherwise it's input_path
        samples_for_norm, sr_for_norm = sf.read(str(current_samples_path), dtype='float32', always_2d=False)
        
        # Ensure mono for audiomentations (ffmpeg resampling should have handled this if it ran)
        if samples_for_norm.ndim > 1 and samples_for_norm.shape[-1] > 1:
             samples_for_norm = np.mean(samples_for_norm, axis=-1)
             log.debug(f"Mixed {current_samples_path.name} to mono for audiomentations step.")

        # Corrected audiomentations parameters
        augmenter = Compose([
            LoudnessNormalization(min_lufs=target_lufs, max_lufs=target_lufs, p=1.0)
        ])
        # Apply augmentation using the sample rate of the (potentially resampled) input
        normalized_samples = augmenter(samples=samples_for_norm, sample_rate=sr_for_norm)
        
        # Save LUFS-normalized samples to a temporary file for ffmpeg peak limiting
        temp_lufs_normalized_path = output_path.with_name(output_path.stem + "_temp_lufs_normalized.wav")
        # Write with sr_for_norm initially, ffmpeg will ensure final target_sr
        sf.write(str(temp_lufs_normalized_path), normalized_samples, sr_for_norm, subtype='PCM_16') 
        log.debug(f"Saved LUFS-normalized audio to temporary file: {temp_lufs_normalized_path} at {sr_for_norm}Hz")

        # Apply True Peak Limiting using ffmpeg to the temporary file
        target_true_peak_db = -1.5
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", str(temp_lufs_normalized_path),
            "-af", f"alimiter=limit={target_true_peak_db}dB:level=false:level_out_fixed=true",
            "-ar", str(target_sr), # Ensure final output sample rate is target_sr
            str(output_path)      # Final output path
        ]

        description = f"True peak limiting ({target_true_peak_db}dBTP) for {temp_lufs_normalized_path.name}"
        if run_ffmpeg_command_core(ffmpeg_cmd, description):
            log.info(f"Successfully normalized and peak limited {input_path.name} to {output_path.name} at {target_sr}Hz")
        else:
            log.error(f"Failed to apply true peak limiting to {temp_lufs_normalized_path.name}. Output may not be peak limited or at target SR.")
            if temp_lufs_normalized_path.exists() and temp_lufs_normalized_path != output_path:
                 shutil.copy2(temp_lufs_normalized_path, output_path)
                 log.warning(f"Copied {temp_lufs_normalized_path.name} to {output_path.name} as fallback (may not be peak limited or at target SR).")
        
        # Clean up temporary files
        if temp_lufs_normalized_path.exists():
            try:
                os.remove(temp_lufs_normalized_path)
                log.debug(f"Removed temporary LUFS file: {temp_lufs_normalized_path}")
            except OSError as e:
                log.warning(f"Could not remove temporary LUFS file {temp_lufs_normalized_path}: {e}")
        if temp_resampled_path and temp_resampled_path.exists():
            try:
                os.remove(temp_resampled_path)
                log.debug(f"Removed temporary resampled file: {temp_resampled_path}")
            except OSError as e:
                log.warning(f"Could not remove temporary resampled file {temp_resampled_path}: {e}")
        
        return output_path

    except Exception as e:
        log.error(f"Error during normalization of {input_path.name}: {e}", exc_info=True)
        # Fallback: copy original if all else fails, to ensure an output file exists if possible
        if input_path != output_path: 
            try:
                shutil.copy2(input_path, output_path)
                log.warning(f"Fallback: Copied original {input_path.name} to {output_path.name} due to normalization error.")
            except Exception as e_copy:
                log.error(f"Fallback copy also failed for {input_path.name}: {e_copy}")
                return None # Critical failure
        return output_path # Return original if copied, or if input_path was same as output_path


def run_generic_preprocess(input_audio_path_str: str, base_output_dir_str: str, pii_safe_file_prefix: str, config: dict = None):
    """
    Runs a generic audio preprocessing pipeline:
    1. Resample original to target sample rate and mono (optional, if needed before separation or as a direct output).
    2. Separate stems (vocals/instrumental).
    3. Normalize each stem to a target LUFS and ensure it's at target_sr & mono.
    
    Accepts pii_safe_file_prefix for all output naming.
    Returns a dictionary with paths to processed stems and potentially the resampled original.
    """
    # --- Debug flags ---
    DEBUG_SKIP_STEM_SEPARATION = False  # Set to True to skip stem separation
    DEBUG_SKIP_VOCAL_NORMALIZATION = False # Set to True to skip vocal normalization
    DEBUG_SKIP_INSTRUMENTAL_NORMALIZATION = False # Set to True to skip instrumental normalization

    log.info(f"Debug Flags: SkipSeparation={DEBUG_SKIP_STEM_SEPARATION}, SkipVocalNorm={DEBUG_SKIP_VOCAL_NORMALIZATION}, SkipInstrumentalNorm={DEBUG_SKIP_INSTRUMENTAL_NORMALIZATION}")

    if not all([Compose, LoudnessNormalization, sf, np]):
        log.warning("Audiomentations components not available. Skipping preprocessing.")
        return None

    input_audio_path = Path(input_audio_path_str)
    base_output_dir = Path(base_output_dir_str)
    if config is None: config = {}

    log.info(f"Original audio: {input_audio_path}")
    log.info(f"Base output directory: {base_output_dir}")
    log.info(f"PII-Safe File Prefix: {pii_safe_file_prefix}")
    log.info(f"Configuration: {config}")

    # --- Create a temporary working directory for this specific preprocessing task ---
    # This temp dir will be within the stage's output dir (base_output_dir)
    # All intermediate files for this one input_audio_path will go here.
    # The pii_safe_file_prefix makes it unique for this input file processing run.
    temp_processing_sub_dir = base_output_dir / f"temp_preprocess_{pii_safe_file_prefix}"
    try:
        temp_processing_sub_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.error(f"Failed to create temporary processing sub-directory {temp_processing_sub_dir}: {e}")
        return None # Cannot proceed

    # Dictionary to store paths of final processed files for return
    processed_files = {}
    target_sr_hz = config.get('target_sr_hz', DEFAULT_TARGET_SR_CORE)

    # --- 1. Optional: Resample original audio to target SR and mono for certain use cases ---
    # This might be useful if a mono, resampled version of the *original* (pre-separation) is needed.
    # If separation is always done, this specific version might be redundant with summing stems later.
    # For now, let's assume it is NOT a primary output unless specifically configured.
    # If it were, its name should use pii_safe_file_prefix.
    # Example: original_mono_resampled_path = base_output_dir / f"{pii_safe_file_prefix}_original_mono_{target_sr_hz}Hz.wav"
    # resample_audio_ffmpeg_core(input_audio_path, original_mono_resampled_path, target_sr_hz, mono=True)
    # processed_files['original_mono_resampled'] = str(original_mono_resampled_path) 

    # --- 2. Separate Stems (vocals/instrumental) --- 
    # Stems will be raw, at original SR of separator's output, potentially stereo (though usually mono)
    raw_stems_output_dir = temp_processing_sub_dir # Place raw stems from separator in a sub-temp dir
    
    vocals_raw_path = None
    instrumental_raw_path = None

    if not DEBUG_SKIP_STEM_SEPARATION:
        initial_separated_stems = separate_stems_core(
            input_audio_path, 
            raw_stems_output_dir, # This is base_temp_dir for separate_stems_core
            config.get('separator_model_core', DEFAULT_SEPARATOR_MODEL_CORE),
            pii_safe_file_prefix, # Pass the prefix down
            config=config # Pass full config for dummy generation SR if needed
        )
        if not initial_separated_stems or 'vocals' not in initial_separated_stems:
            log.error("Stem separation failed to produce vocals. Preprocessing aborted (or will use dummies if other steps active).")
            # Fall through to dummy creation if normalization steps are not skipped
        else:
            vocals_raw_path = initial_separated_stems['vocals']
            instrumental_raw_path = initial_separated_stems.get('instrumental')
    else:
        log.info("DEBUG: Skipping stem separation.")

    # Create dummy raw stems if separation was skipped or failed, and normalization is expected
    if vocals_raw_path is None or not vocals_raw_path.exists():
        # Create a dummy vocal file if vocal normalization is not skipped or if instrumental norm is not skipped (implies need for full structure)
        if not DEBUG_SKIP_VOCAL_NORMALIZATION or not DEBUG_SKIP_INSTRUMENTAL_NORMALIZATION:
            dummy_vocal_name = f"{pii_safe_file_prefix}_vocals_raw_dummy.wav"
            vocals_raw_path = raw_stems_output_dir / dummy_vocal_name
            try:
                if np and sf: # Create a valid silent WAV
                    sf.write(str(vocals_raw_path), np.zeros(int(1 * target_sr_hz), dtype='float32'), target_sr_hz)
                    log.info(f"Created dummy silent raw vocal stem (1s at {target_sr_hz}Hz): {vocals_raw_path}")
                else: # Fallback to touch
                    vocals_raw_path.touch()
                    log.info(f"Created empty dummy raw vocal stem (fallback): {vocals_raw_path}")
            except Exception as e_dummy:
                log.error(f"Failed to create dummy raw vocal stem {vocals_raw_path}: {e_dummy}")
                # If we can't even make a dummy, and it's needed, we might have issues.

    if instrumental_raw_path is None or not instrumental_raw_path.exists():
        if not DEBUG_SKIP_INSTRUMENTAL_NORMALIZATION:
            dummy_instr_name = f"{pii_safe_file_prefix}_instrumental_raw_dummy.wav"
            instrumental_raw_path = raw_stems_output_dir / dummy_instr_name
            try:
                if np and sf: # Create a valid silent WAV
                    sf.write(str(instrumental_raw_path), np.zeros(int(1 * target_sr_hz), dtype='float32'), target_sr_hz)
                    log.info(f"Created dummy silent raw instrumental stem (1s at {target_sr_hz}Hz): {instrumental_raw_path}")
                else: # Fallback to touch
                    instrumental_raw_path.touch()
                    log.info(f"Created empty dummy raw instrumental stem (fallback): {instrumental_raw_path}")
            except Exception as e_dummy:
                log.error(f"Failed to create dummy raw instrumental stem {instrumental_raw_path}: {e_dummy}")


    # --- 3. Normalize Vocal Stem ---
    if not DEBUG_SKIP_VOCAL_NORMALIZATION:
        if vocals_raw_path and vocals_raw_path.exists():
            # Output normalized vocals directly to the main stage output directory (base_output_dir)
            normalized_vocals_filename = f"{pii_safe_file_prefix}_vocals_normalized.wav"
            normalized_vocals_path = base_output_dir / normalized_vocals_filename 
            
            processed_vocals = normalize_stem_audiomentations_core(
                vocals_raw_path, 
                normalized_vocals_path, 
                config.get('vocals_lufs', DEFAULT_VOCAL_TARGET_LUFS_CORE),
                target_sr_hz
            )
            if processed_vocals:
                processed_files['vocals_normalized'] = str(processed_vocals)
                log.info(f"Processed vocal stem saved to: {processed_vocals}")
            else:
                log.warning(f"Vocal stem processing failed for {vocals_raw_path}. It will not be included in outputs.")
        else:
            log.warning("Raw vocal stem not available or not found. Skipping vocal normalization.")
    else:
        log.info("DEBUG: Skipping vocal normalization.")
        # If skipped, still provide a path for consistency if the raw/dummy file exists
        if vocals_raw_path and vocals_raw_path.exists():
            # Copy raw/dummy to expected output location, or just record its path if it's already in temp_processing_sub_dir
            final_vocal_path = base_output_dir / f"{pii_safe_file_prefix}_vocals_skipped_norm.wav"
            shutil.copy2(vocals_raw_path, final_vocal_path)
            processed_files['vocals_normalized'] = str(final_vocal_path)
            log.info(f"DEBUG: Using raw/dummy vocal stem as output (normalization skipped): {final_vocal_path}")
        else:
            log.warning("DEBUG: Vocal normalization skipped, and no raw/dummy vocal stem available to use as output.")

    # --- 4. Normalize Instrumental Stem ---
    if not DEBUG_SKIP_INSTRUMENTAL_NORMALIZATION:
        if instrumental_raw_path and instrumental_raw_path.exists():
            # Output normalized instrumentals directly to the main stage output directory (base_output_dir)
            normalized_instrumental_filename = f"{pii_safe_file_prefix}_instrumental_normalized.wav"
            normalized_instrumental_path = base_output_dir / normalized_instrumental_filename

            processed_instrumental = normalize_stem_audiomentations_core(
                instrumental_raw_path, 
                normalized_instrumental_path, 
                config.get('instrumental_lufs', DEFAULT_INSTRUMENTAL_TARGET_LUFS_CORE),
                target_sr_hz
            )
            if processed_instrumental:
                processed_files['instrumental_normalized'] = str(processed_instrumental)
                log.info(f"Processed instrumental stem saved to: {processed_instrumental}")
            else:
                log.warning(f"Instrumental stem processing failed for {instrumental_raw_path}. It will not be included in outputs.")
        else:
            log.warning("Raw instrumental stem not available or not found. Skipping instrumental normalization.")
    else:
        log.info("DEBUG: Skipping instrumental normalization.")
        if instrumental_raw_path and instrumental_raw_path.exists():
            final_instr_path = base_output_dir / f"{pii_safe_file_prefix}_instrumental_skipped_norm.wav"
            shutil.copy2(instrumental_raw_path, final_instr_path)
            processed_files['instrumental_normalized'] = str(final_instr_path)
            log.info(f"DEBUG: Using raw/dummy instrumental stem as output (normalization skipped): {final_instr_path}")
        else:
            log.warning("DEBUG: Instrumental normalization skipped, and no raw/dummy instrumental stem available to use as output.")

    # --- 5. Cleanup temporary processing sub-directory ---
    if temp_processing_sub_dir.exists():
        try:
            shutil.rmtree(temp_processing_sub_dir)
            log.info(f"Cleaned up temporary processing sub-directory: {temp_processing_sub_dir}")
        except Exception as e_cleanup:
            log.warning(f"Failed to cleanup temporary processing sub-directory {temp_processing_sub_dir}: {e_cleanup}")

    if not processed_files:
        log.error("Audio preprocessing resulted in no usable output files.")
        return None
        
    log.info(f"Audio preprocessing complete. Returning: {json.dumps(processed_files, indent=2)}")
    return processed_files # This dict should contain PII-safe paths


if __name__ == "__main__":
    # Example Usage for audio_preprocessor.py
    # Ensure the logger used in this script is configured by calling setup_logging_core()
    # If this script is run directly, the module-level log = setup_logging_core() already does this.
    # If functions are imported, the importing module should set up logging for its own use or configure root.
    
    example_input_filename_core = "example_preprocessor_input.wav"
    test_input_file_core = Path(example_input_filename_core)

    if not test_input_file_core.exists():
        log.warning(f"Test input file '{example_input_filename_core}' not found.")
        log.warning("Attempting to create a simple dummy WAV file for testing.")
        if sf and np:
            try:
                samplerate = 44100; duration = 2; frequency = 440
                t = np.linspace(0, duration, int(samplerate * duration), False)
                note = 0.4 * np.sin(2 * np.pi * frequency * t) + 0.2 * np.sin(2 * np.pi * (frequency * 2.5) * t)
                sf.write(test_input_file_core, note, samplerate)
                log.info(f"Created dummy test file: {test_input_file_core}")
            except Exception as e_create_dummy:
                 log.error(f"Could not create dummy test file: {e_create_dummy}. Please provide a '{example_input_filename_core}'.")
                 exit()
        else:
            log.error(f"Soundfile/numpy not available to create a dummy file. Exiting test for {__file__}.")
            exit()

    if test_input_file_core.exists():
        output_directory_core = "preprocessed_outputs"
        
        test_config_core = {
            'separator_model_core': DEFAULT_SEPARATOR_MODEL_CORE,
            'vocals_lufs': -6.0, # Example override
            'instrumental_lufs': -12.0, # Example override
            'target_sr_hz': 44100
        }
        
        log.info(f"Running generic preprocessor with test config on: {test_input_file_core}")
        result_paths = run_generic_preprocess(str(test_input_file_core), output_directory_core, "test_prefix", config=test_config_core)
        
        if result_paths:
            log.info("Preprocessing finished. Resulting stem paths:")
            for stem_type, path in result_paths.items():
                log.info(f"  {stem_type}: {path}")
        else:
            log.error("Generic preprocessing failed.")
    else:
        log.error(f"Test input file '{example_input_filename_core}' still not found. Cannot run example for {__file__}.") 