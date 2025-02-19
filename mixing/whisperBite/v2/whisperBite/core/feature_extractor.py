"""
Audio feature extraction module for WhisperBite.
Handles extraction of audio features needed for processing.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from pydub import AudioSegment
from config import AudioProcessingError

@dataclass
class AudioFeatures:
    """Container for extracted audio features."""
    vad_segments: List[Tuple[float, float]]
    pitch_profile: np.ndarray
    energy_profile: np.ndarray
    duration: float

class FeatureExtractor:
    def __init__(self):
        self.window_size = 2048
        self.hop_length = 512
        
    def extract_features(self, audio_path: str) -> AudioFeatures:
        """Extract all necessary features from an audio file."""
        try:
            # Load audio
            audio = AudioSegment.from_wav(audio_path)
            samples = np.array(audio.get_array_of_samples())
            
            # Extract features
            energy = self._compute_energy_profile(samples)
            vad_segments = self._detect_voice_activity(energy)
            pitch = self._extract_pitch_features(samples)
            
            return AudioFeatures(
                vad_segments=vad_segments,
                pitch_profile=pitch,
                energy_profile=energy,
                duration=len(audio) / 1000.0
            )
            
        except Exception as e:
            raise AudioProcessingError(f"Feature extraction failed: {str(e)}")

    def _compute_energy_profile(self, samples: np.ndarray) -> np.ndarray:
        """Compute energy profile of the audio."""
        energy = []
        for i in range(0, len(samples), self.hop_length):
            window = samples[i:i + self.window_size]
            if len(window) == self.window_size:  # Only process full windows
                energy.append(np.sum(window ** 2))
        return np.array(energy)

    def _detect_voice_activity(self, energy: np.ndarray) -> List[Tuple[float, float]]:
        """Detect voice activity segments based on energy profile."""
        threshold = np.mean(energy) * 0.5
        segments = []
        in_speech = False
        start = 0
        
        for i, e in enumerate(energy):
            if not in_speech and e > threshold:
                start = i
                in_speech = True
            elif in_speech and e <= threshold:
                # Convert frame indices to time
                start_time = start * self.hop_length / 44100
                end_time = i * self.hop_length / 44100
                segments.append((start_time, end_time))
                in_speech = False
                
        return segments

    def _extract_pitch_features(self, samples: np.ndarray) -> np.ndarray:
        """Extract pitch-related features using zero-crossing rate."""
        zcr = []
        for i in range(0, len(samples) - self.window_size, self.hop_length):
            window = samples[i:i + self.window_size]
            # Count zero crossings
            crossings = np.sum(np.diff(np.signbit(window)))
            zcr.append(crossings / self.window_size)
        return np.array(zcr)

    def merge_short_segments(self, segments: List[Tuple[float, float]], 
                           min_duration: float = 0.5) -> List[Tuple[float, float]]:
        """Merge segments that are too short."""
        if not segments:
            return []
            
        merged = []
        current_start, current_end = segments[0]
        
        for start, end in segments[1:]:
            if start - current_end < min_duration:
                # Merge with current segment
                current_end = end
            else:
                # Add current segment if it's long enough
                if current_end - current_start >= min_duration:
                    merged.append((current_start, current_end))
                current_start, current_end = start, end
                
        # Add the last segment if it's long enough
        if current_end - current_start >= min_duration:
            merged.append((current_start, current_end))
            
        return merged
