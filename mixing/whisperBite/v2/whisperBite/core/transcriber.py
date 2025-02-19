"""
Whisper transcription handler for WhisperBite.
"""

import logging
from typing import Dict, Optional
import whisper
import torch
import numpy as np
from whisperBite.config import AudioProcessingError

logger = logging.getLogger(__name__)

class WhisperTranscriber:
    """Handles Whisper transcription operations."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}  # Cache for loaded models
        logger.info(f"Initializing WhisperTranscriber with device: {self.device}")
    
    def transcribe(self, audio_path: str, model_name: str = "base") -> Dict:
        """
        Transcribe audio using Whisper with word timestamps.
        Returns dictionary with transcription results.
        """
        try:
            # Get or load model
            model = self._get_model(model_name)
            
            # Set options to get word timestamps
            options = {
                'language': None,  # Auto-detect language
                'task': 'transcribe',
                'word_timestamps': True,  # Enable word timestamps
                'verbose': None,
            }
            
            # Perform transcription
            result = model.transcribe(audio_path, **options)
            
            # Log debug info about segments and words
            total_words = sum(len(seg.get('words', [])) for seg in result.get('segments', []))
            logger.info(f"Transcription produced {len(result.get('segments', []))} segments and {total_words} words")
            
            # Process and validate results
            processed_result = self._process_transcription(result)
            return processed_result
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise AudioProcessingError(f"Transcription failed: {str(e)}")
    
    def _get_model(self, model_name: str) -> whisper.Whisper:
        """Get cached model or load if not exists."""
        if model_name not in self.models:
            logger.info(f"Loading Whisper model: {model_name}")
            try:
                self.models[model_name] = whisper.load_model(model_name).to(self.device)
            except Exception as e:
                raise AudioProcessingError(f"Failed to load Whisper model: {str(e)}")
        return self.models[model_name]
    
    def _process_transcription(self, result: Dict) -> Dict:
        """Process and validate transcription results."""
        processed = {
            'text': result.get('text', '').strip(),
            'language': result.get('language', 'unknown'),
            'segments': []
        }
        
        # Process segments
        for segment in result.get('segments', []):
            processed_segment = {
                'start': float(segment['start']),
                'end': float(segment['end']),
                'text': segment['text'].strip(),
                'words': []
            }
            
            # Process word timestamps if available
            if 'words' in segment:
                for word in segment['words']:
                    # Handle both 'word' and 'text' keys that Whisper might use
                    word_text = word.get('word', word.get('text', '')).strip()
                    processed_segment['words'].append({
                        'text': word_text,
                        'start': float(word['start']),
                        'end': float(word['end']),
                        'probability': float(word.get('probability', 0.0))
                    })
            
            if self._validate_segment(processed_segment):
                processed['segments'].append(processed_segment)
                
        # Log some examples for debugging
        if processed['segments']:
            first_seg = processed['segments'][0]
            if first_seg.get('words'):
                logger.info(f"Example processed words: {first_seg['words'][:3]}")
        
        return processed
    
    def _validate_segment(self, segment: Dict) -> bool:
        """Validate a transcription segment."""
        required_keys = {'start', 'end', 'text'}
        if not all(key in segment for key in required_keys):
            logger.warning(f"Missing required keys in segment: {segment}")
            return False
            
        if segment['end'] <= segment['start']:
            logger.warning(f"Invalid segment timing: {segment}")
            return False
            
        if not segment['text'].strip():
            logger.warning("Empty segment text")
            return False
            
        return True
    
    def cleanup(self):
        """Clean up loaded models."""
        logger.info("Cleaning up Whisper models")
        for model in self.models.values():
            del model
        self.models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()