import logging
from pathlib import Path
import os
import torch
from pyannote.audio import Pipeline

logger = logging.getLogger(__name__)

# Simple replica of WhisperBite's format_speaker_label, 
# though for RTTM, raw pyannote labels like 'SPEAKER_00' are fine.
# This might be useful if a specific 'S0', 'S1' format is desired later.
# def format_pyannote_label(label_from_pyannote: str) -> str:
#     try:
#         return f"S{int(label_from_pyannote.split('_')[-1])}"
#     except: # noqa
#         return label_from_pyannote # Fallback

def run_diarization(vocal_stem_path: str, output_dir_str: str, pii_safe_file_prefix: str, config: dict = None):
    """
    Performs speaker diarization using pyannote.audio.
    Outputs an RTTM file.
    Relies on the HF_TOKEN environment variable being set externally.
    pii_safe_file_prefix is received but not explicitly used for naming if vocal_stem_path is already correctly prefixed.
    """
    if vocal_stem_path is None:
        logger.error("[Pyannote Diarization] Critical error: vocal_stem_path is None. Cannot proceed with diarization.")
        return None

    if config is None:
        config = {}

    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"[Pyannote Diarization] Processing: {vocal_stem_path}")
    logger.info(f"[Pyannote Diarization] Config (for model name, num_speakers etc.): {config}")

    if not os.environ.get('HF_TOKEN'):
        logger.warning("[Pyannote Diarization] Warning: HF_TOKEN environment variable not found. Diarization might fail or use limited models if the chosen pipeline requires authentication.")
        # Proceeding, pyannote will handle the error if a token is strictly necessary for the chosen model.
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[Pyannote Diarization] Using device: {device}")

        diarization_model_name = config.get("diarization_model_name", "pyannote/speaker-diarization-3.1")
        logger.info(f"[Pyannote Diarization] Loading pipeline: {diarization_model_name}")
        
        try:
            # pyannote.audio will use HF_TOKEN from environment if set and required by the model
            pipeline = Pipeline.from_pretrained(diarization_model_name)
            pipeline.to(device)
        except Exception as e:
            logger.error(f"[Pyannote Diarization] Failed to load diarization pipeline '{diarization_model_name}': {e}")
            logger.error(f"[Pyannote Diarization] Ensure HF_TOKEN is correctly set in your environment if model requires auth, and that you have internet access.")
            return None

        # Determine speaker count arguments for the pipeline call
        pipeline_kwargs = {}
        speaker_args_provided = False
        
        if 'num_speakers' in config and config['num_speakers'] is not None:
            pipeline_kwargs['num_speakers'] = config['num_speakers']
            speaker_args_provided = True
            logger.info(f"[Pyannote Diarization] Using configured num_speakers: {config['num_speakers']}")
        
        if 'min_speakers' in config and config['min_speakers'] is not None:
            pipeline_kwargs['min_speakers'] = config['min_speakers']
            speaker_args_provided = True
            logger.info(f"[Pyannote Diarization] Using configured min_speakers: {config['min_speakers']}")

        if 'max_speakers' in config and config['max_speakers'] is not None:
            pipeline_kwargs['max_speakers'] = config['max_speakers']
            speaker_args_provided = True
            logger.info(f"[Pyannote Diarization] Using configured max_speakers: {config['max_speakers']}")

        if not speaker_args_provided:
            logger.info("[Pyannote Diarization] No speaker count (num_speakers, min_speakers, max_speakers) provided in config. Pipeline will attempt to auto-detect speaker count.")

        logger.info(f"[Pyannote Diarization] Running diarization with arguments: {pipeline_kwargs} on {vocal_stem_path}")
        
        try:
            # Ensure vocal_stem_path is an absolute path string for the pipeline
            resolved_vocal_stem_path = str(Path(vocal_stem_path).resolve())
            diarization_annotation = pipeline(resolved_vocal_stem_path, **pipeline_kwargs)
        except Exception as e:
            logger.error(f"[Pyannote Diarization] Pipeline execution failed on '{resolved_vocal_stem_path}': {e}")
            return None

        base_name = Path(vocal_stem_path).stem
        rttm_file_path = output_dir / f"{base_name}_diarization.rttm"
        
        logger.info(f"[Pyannote Diarization] Writing RTTM output to: {rttm_file_path}")
        with open(rttm_file_path, 'w') as f_rttm:
            file_id = base_name # RTTM file_id is typically the original audio file's identifier
            for turn, _, speaker_label in diarization_annotation.itertracks(yield_label=True):
                start_time = turn.start
                duration = turn.end - start_time
                # RTTM format: SPEAKER <file_id> <channel> <start_time> <duration> <NA> <NA> <speaker_id> <NA> <NA>
                f_rttm.write(f"SPEAKER {file_id} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {speaker_label} <NA> <NA>\n")
        
        logger.info(f"[Pyannote Diarization] RTTM file successfully saved to: {rttm_file_path}")
        return {"rttm_file_path": str(rttm_file_path)}

    except Exception as general_error:
        logger.error(f"[Pyannote Diarization] An unexpected error occurred during diarization: {general_error}")
        return None 