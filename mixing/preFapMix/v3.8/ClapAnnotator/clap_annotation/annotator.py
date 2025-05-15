import logging
from pathlib import Path
from typing import List, Dict, Union, Optional

import torch
import numpy as np
import librosa

# Attempt to import transformers components
try:
    from transformers import AutoProcessor, AutoModel
except ImportError:
    AutoProcessor = None
    AutoModel = None
    logging.error("Failed to import AutoProcessor or AutoModel from transformers. Please ensure the library is installed.")

from config import settings

log = logging.getLogger(__name__)

class CLAPAnnotatorWrapper:
    """Wraps the CLAP model for audio annotation with chunking."""

    def __init__(self,
                 model_name: str = settings.CLAP_MODEL_NAME,
                 chunk_duration_s: int = settings.CLAP_CHUNK_DURATION_S,
                 expected_sr: int = settings.CLAP_EXPECTED_SR,
                 device: Optional[str] = None):
        """Initializes the CLAP annotator.

        Args:
            model_name: Name of the CLAP model on Hugging Face Hub.
            chunk_duration_s: Duration of audio chunks for processing (seconds).
            expected_sr: Target sample rate required by the CLAP model.
            device: Device to run inference on ("cuda", "cpu", or None for auto-detect).
        """
        if AutoProcessor is None or AutoModel is None:
            raise RuntimeError("transformers library not found or failed to import.")

        self.model_name = model_name
        self.chunk_duration_s = chunk_duration_s
        self.expected_sr = expected_sr

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Using device: {self.device}")

        self.processor = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads the CLAP model and processor."""
        log.info(f"Loading CLAP model: {self.model_name}")
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval() # Set model to evaluation mode
            log.info("CLAP model loaded successfully.")
        except Exception as e:
            log.exception(f"Failed to load CLAP model {self.model_name}: {e}")
            raise RuntimeError(f"Failed to load CLAP model: {e}") from e

    @torch.no_grad() # Disable gradient calculation for inference
    def annotate(self, 
                 audio_path: Path, 
                 text_prompts: List[str], 
                 confidence_threshold: float = settings.DEFAULT_CLAP_CONFIDENCE_THRESHOLD,
                 progress_callback: Optional[callable] = None) -> Dict[str, Union[str, List[Dict]]]:
        """Annotates a single audio file with given text prompts using chunking.

        Args:
            audio_path: Path to the resampled audio file (at expected_sr, mono).
            text_prompts: A list of text prompts to search for.
            confidence_threshold: Minimum probability score to include a detection.
            progress_callback: Optional function to call with progress updates (e.g., progress_callback(current_chunk, total_chunks)).

        Returns:
            A dictionary containing the analysis results for the audio file:
            {
                "file": str(audio_path),
                "detections": [
                    {"start_time_s": float, "end_time_s": float, "prompts": {prompt: probability}}
                ]
            }

        Raises:
            FileNotFoundError: If the audio file does not exist.
            ValueError: If text_prompts list is empty.
            RuntimeError: If audio loading or model inference fails.
        """
        if not audio_path.is_file():
            log.error(f"Input audio file for annotation not found: {audio_path}")
            raise FileNotFoundError(f"Input audio file not found: {audio_path}")
        if not text_prompts:
             log.error("Cannot annotate: text_prompts list is empty.")
             raise ValueError("text_prompts list cannot be empty.")

        log.info(f"Starting CLAP annotation for: {audio_path} with {len(text_prompts)} prompts.")
        
        # Use relative path instead of absolute path
        try:
            relative_path = audio_path.relative_to(settings.PROJECT_ROOT)
            results = {"file": str(relative_path), "detections": []}
        except ValueError:
            # If the path cannot be made relative (different drive, etc.), use the filename only
            results = {"file": str(audio_path.name), "detections": []}

        # 1. Pre-calculate text features
        try:
            log.debug("Processing text prompts...")
            text_inputs = self.processor(text=text_prompts, return_tensors="pt", padding=True).to(self.device)
            text_features = self.model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True) # Normalize features
            log.debug("Text features calculated.")
        except Exception as e:
            log.exception("Failed to process text prompts with CLAP processor/model.")
            raise RuntimeError("Failed during text feature extraction.") from e

        # 2. Load and chunk audio
        try:
            log.debug(f"Loading audio file: {audio_path} with sr={self.expected_sr}")
            # Ensure mono=True, although resampling step should have handled this
            waveform, sr = librosa.load(str(audio_path), sr=self.expected_sr, mono=True) 
            log.debug(f"Audio loaded. Shape: {waveform.shape}, Sample Rate: {sr}")
            if sr != self.expected_sr:
                 log.warning(f"Audio sample rate ({sr}) does not match expected ({self.expected_sr}) after loading. Check resampling step.")
        except Exception as e:
            log.exception(f"Failed to load audio file: {audio_path}")
            raise RuntimeError(f"Failed to load audio: {e}") from e
        
        total_samples = waveform.shape[0]
        chunk_samples = self.expected_sr * self.chunk_duration_s
        num_chunks = int(np.ceil(total_samples / chunk_samples))
        log.info(f"Processing audio in {num_chunks} chunks of approx {self.chunk_duration_s}s...")

        # 3. Process each chunk
        for i in range(num_chunks):
            if progress_callback:
                progress_callback(i + 1, num_chunks) # Update progress (1-based index)

            start_sample = i * chunk_samples
            end_sample = min((i + 1) * chunk_samples, total_samples)
            audio_chunk = waveform[start_sample:end_sample]
            
            start_time_s = start_sample / self.expected_sr
            end_time_s = end_sample / self.expected_sr
            log.debug(f"Processing chunk {i+1}/{num_chunks} ({start_time_s:.2f}s - {end_time_s:.2f}s)")

            # Check if chunk is too short (e.g., less than a few ms)
            min_chunk_samples = 50 # Arbitrary minimum sample count, adjust if needed
            if len(audio_chunk) < min_chunk_samples:
                log.warning(f"Skipping chunk {i+1} as it is too short ({len(audio_chunk)} samples).")
                continue

            try:
                # Process audio chunk
                # Using np array directly often works, but check docs if issues arise
                audio_inputs = self.processor(audios=audio_chunk, sampling_rate=self.expected_sr, return_tensors="pt").to(self.device)
                audio_features = self.model.get_audio_features(**audio_inputs)
                audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True) # Normalize
                
                # Calculate similarity (logits)
                logits_per_audio = audio_features @ text_features.T # Matmul for similarity
                
                # Apply sigmoid to get probabilities [0, 1]
                probs = torch.sigmoid(logits_per_audio).squeeze().cpu().numpy()
                
                # Create dictionary of prompts and their probabilities for this chunk
                chunk_probabilities = {prompt: float(prob) for prompt, prob in zip(text_prompts, probs)}
                log.debug(f"Chunk {i+1} probabilities: {chunk_probabilities}")
                
                # Filter based on confidence threshold and add to results
                filtered_prompts = {p: prob for p, prob in chunk_probabilities.items() if prob >= confidence_threshold}
                
                if filtered_prompts:
                    results["detections"].append({
                        "start_time_s": round(start_time_s, 3),
                        "end_time_s": round(end_time_s, 3),
                        "prompts": filtered_prompts
                    })
                    log.debug(f"Added detections above threshold {confidence_threshold} for chunk {i+1}")

            except torch.cuda.OutOfMemoryError as oom_error: # Catch OOM specifically
                log.exception(f"CUDA Out of Memory error during CLAP annotation chunk {i+1}. Try a smaller chunk size or less demanding models.")
                raise RuntimeError("CUDA Out of Memory during CLAP annotation.") from oom_error
            except Exception as e:
                log.exception(f"Error processing CLAP annotation chunk {i+1} ({start_time_s:.2f}s - {end_time_s:.2f}s): {e}")
                # Continue to next chunk, but log the error
                # Optionally, add an error marker to the results for this timeframe?
                # results["detections"].append({"start_time_s": start_time_s, "end_time_s": end_time_s, "error": str(e)})

        log.info(f"CLAP annotation finished for: {audio_path}. Found {len(results['detections'])} detections above threshold.")
        return results 