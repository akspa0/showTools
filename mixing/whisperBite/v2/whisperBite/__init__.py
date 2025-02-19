"""
WhisperBite - Advanced audio processing and transcription tool.
"""

from .processor import AudioProcessor
from .config import ProcessingOptions, AudioProcessingError

__version__ = "1.0.0"
__all__ = ['AudioProcessor', 'ProcessingOptions', 'AudioProcessingError']
