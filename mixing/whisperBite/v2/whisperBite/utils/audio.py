"""
Audio utilities for WhisperBite.
"""

import os
import logging
from typing import List
from pydub import AudioSegment
from ..config import AudioProcessingError, SUPPORTED_FORMATS

logger = logging.getLogger(__name__)

class AudioUtils:
    @staticmethod
    def validate_audio_file(file_path: str) -> bool:
        """Validate audio file format and accessibility."""
        if not os.path.exists(file_path):
            return False
        
        return any(file_path.lower().endswith(ext) for ext in SUPPORTED_FORMATS)

    @staticmethod
    def get_audio_files(input_path: str) -> List[str]:
        """Get list of supported audio files from directory."""
        if os.path.isfile(input_path):
            if AudioUtils.validate_audio_file(input_path):
                return [input_path]
            else:
                raise AudioProcessingError(f"Unsupported file format: {input_path}")
        
        audio_files = []
        for root, _, files in os.walk(input_path):
            for file in files:
                full_path = os.path.join(root, file)
                if AudioUtils.validate_audio_file(full_path):
                    audio_files.append(full_path)
        
        return audio_files

    @staticmethod
    def get_audio_duration(file_path: str) -> float:
        """Get duration of audio file in seconds."""
        try:
            audio = AudioSegment.from_file(file_path)
            return len(audio) / 1000.0
        except Exception as e:
            raise AudioProcessingError(f"Failed to get audio duration: {str(e)}")

    @staticmethod
    def get_audio_info(file_path: str) -> dict:
        """Get detailed information about audio file."""
        try:
            audio = AudioSegment.from_file(file_path)
            return {
                'duration': len(audio) / 1000.0,
                'sample_rate': audio.frame_rate,
                'channels': audio.channels,
                'bit_depth': audio.sample_width * 8,
                'format': os.path.splitext(file_path)[1][1:],
                'size_bytes': os.path.getsize(file_path)
            }
        except Exception as e:
            raise AudioProcessingError(f"Failed to get audio info: {str(e)}")
