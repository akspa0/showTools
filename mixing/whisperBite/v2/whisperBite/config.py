"""
Configuration module for WhisperBite.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProcessingOptions:
    """Configuration options for audio processing."""
    word_level: bool = False
    diarization: bool = False
    vocal_separation: bool = False
    model_name: str = "turbo"  # Changed default from 'base' to 'turbo'
    num_speakers: int = 2
    output_dir: str = "output"
    
    def validate(self) -> bool:
        """Validate configuration options."""
        if self.num_speakers < 1:
            return False
        if self.model_name not in ["base", "small", "medium", "large", "turbo"]:
            return False
        return True

class AudioProcessingError(Exception):
    """Custom exception for audio processing errors."""
    pass

def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers
    )

AUDIO_SETTINGS = {
    'sample_rate': 44100,
    'channels': 1,
    'bit_depth': 16,
    'target_lufs': -16,
    'lra': 7,
    'true_peak': -1.0
}

SUPPORTED_FORMATS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}

CACHE_SETTINGS = {
    'max_size': 1024 * 1024 * 1024,  # 1GB
    'cleanup_threshold': 0.9,  # 90%
    'entry_timeout': 3600  # 1 hour
}
