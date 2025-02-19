"""
WhisperBite processing modules.
"""

from .demucs_processor import DemucsProcessor
from .post_processor import VocalPostProcessor
from .vocal_separator import VocalSeparator

__all__ = ['DemucsProcessor', 'VocalPostProcessor', 'VocalSeparator']
