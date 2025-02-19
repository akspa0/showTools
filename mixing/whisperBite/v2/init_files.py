# whisperBite/__init__.py
"""
WhisperBite - Advanced audio processing and transcription tool.
"""

from .processor import AudioProcessor
from .config import ProcessingOptions, AudioProcessingError

__version__ = "1.0.0"
__all__ = ['AudioProcessor', 'ProcessingOptions', 'AudioProcessingError']

# whisperBite/processors/__init__.py
"""
WhisperBite processing modules.
"""

from .demucs_processor import DemucsProcessor
from .post_processor import VocalPostProcessor
from .vocal_separator import VocalSeparator

__all__ = ['DemucsProcessor', 'VocalPostProcessor', 'VocalSeparator']

# whisperBite/core/__init__.py
"""
WhisperBite core components.
"""

from .normalizer import AudioNormalizer
from .transcriber import WhisperTranscriber
from .diarizer import SpeakerDiarizer
from .word_splitter import WordSplitter
from .feature_extractor import FeatureExtractor

__all__ = [
    'AudioNormalizer',
    'WhisperTranscriber',
    'SpeakerDiarizer',
    'WordSplitter',
    'FeatureExtractor'
]

# whisperBite/utils/__init__.py
"""
WhisperBite utility modules.
"""

from .audio import AudioUtils
from .download import DownloadUtils
from .file_handler import FileHandler

__all__ = ['AudioUtils', 'DownloadUtils', 'FileHandler']