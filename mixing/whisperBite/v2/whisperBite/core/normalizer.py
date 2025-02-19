"""
Audio normalization module for WhisperBite.
Handles all audio normalization and validation.
"""

import os
import subprocess
import logging
from typing import Optional
import numpy as np
from pydub import AudioSegment
from whisperBite.config import AUDIO_SETTINGS, AudioProcessingError

logger = logging.getLogger(__name__)

class AudioNormalizer:
    def __init__(self):
        self.lufs_tolerance = 5.0  # Allow ±5 dB LUFS variation
        self.rate_tolerance = 100  # Allow ±100 Hz sample rate variation

    def normalize(self, input_audio: str) -> str:
        """
        Normalize audio using optimal speech parameters.
        Returns path to normalized audio file.
        """
        try:
            output_dir = os.path.dirname(input_audio)
            output_path = os.path.join(
                output_dir,
                "normalized",
                f"{os.path.splitext(os.path.basename(input_audio))[0]}_normalized.wav"
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            command = [
                "ffmpeg", "-i", input_audio,
                "-af", f"loudnorm=I={AUDIO_SETTINGS['target_lufs']}:"
                       f"LRA={AUDIO_SETTINGS['lra']}:"
                       f"TP={AUDIO_SETTINGS['true_peak']}:"
                       f"print_format=summary",
                "-ar", str(AUDIO_SETTINGS['sample_rate']),
                "-ac", str(AUDIO_SETTINGS['channels']),
                "-c:a", "pcm_s16le",
                output_path
            ]

            # Run FFmpeg
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True
            )

            if not os.path.exists(output_path):
                raise AudioProcessingError("Normalized file was not created")

            # Log the normalization result
            logger.info(f"Normalized audio saved to: {output_path}")
            
            # Validate the normalized audio
            validation_result = self._validate_audio(output_path)
            if not validation_result['valid']:
                logger.warning(f"Audio validation warnings: {validation_result['warnings']}")
            
            return output_path

        except subprocess.CalledProcessError as e:
            raise AudioProcessingError(f"FFmpeg normalization failed: {e.stderr}")
        except Exception as e:
            raise AudioProcessingError(f"Normalization failed: {str(e)}")

    def _validate_audio(self, audio_path: str) -> dict:
        """
        Validate normalized audio meets requirements.
        Returns dict with validation results and warnings.
        """
        try:
            audio = AudioSegment.from_wav(audio_path)
            warnings = []
            
            # Check sample rate
            if abs(audio.frame_rate - AUDIO_SETTINGS['sample_rate']) > self.rate_tolerance:
                warnings.append(
                    f"Sample rate {audio.frame_rate} Hz differs significantly from target "
                    f"{AUDIO_SETTINGS['sample_rate']} Hz"
                )
                
            # Check channels
            if audio.channels != AUDIO_SETTINGS['channels']:
                warnings.append(
                    f"Channel count {audio.channels} differs from target "
                    f"{AUDIO_SETTINGS['channels']}"
                )
                
            # Check bit depth
            if audio.sample_width * 8 != AUDIO_SETTINGS['bit_depth']:
                warnings.append(
                    f"Bit depth {audio.sample_width * 8} differs from target "
                    f"{AUDIO_SETTINGS['bit_depth']}"
                )

            # Check loudness (approximate LUFS)
            target_lufs = AUDIO_SETTINGS['target_lufs']
            current_lufs = audio.dBFS
            if abs(current_lufs - target_lufs) > self.lufs_tolerance:
                warnings.append(
                    f"LUFS {current_lufs} is outside tolerance range of "
                    f"{target_lufs}±{self.lufs_tolerance}"
                )

            # Log validation results
            if warnings:
                for warning in warnings:
                    logger.warning(f"Audio validation warning: {warning}")
            else:
                logger.info("Audio validation passed")

            return {
                'valid': True,  # Always return valid, but with warnings if needed
                'warnings': warnings,
                'info': {
                    'sample_rate': audio.frame_rate,
                    'channels': audio.channels,
                    'bit_depth': audio.sample_width * 8,
                    'lufs': current_lufs,
                    'duration': len(audio) / 1000.0
                }
            }

        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return {
                'valid': False,
                'warnings': [f"Validation failed: {str(e)}"],
                'info': None
            }

    def _calculate_lufs(self, audio: AudioSegment) -> float:
        """
        Calculate LUFS (Loudness Units relative to Full Scale).
        This is an approximation as pydub doesn't directly support LUFS measurement.
        """
        return audio.dBFS  # Using dBFS as an approximation