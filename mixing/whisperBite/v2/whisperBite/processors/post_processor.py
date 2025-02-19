"""
Post-processing module for separated vocals.
Handles audio enhancement and noise reduction.
"""

import logging
import numpy as np
from pydub import AudioSegment
from config import AudioProcessingError, AUDIO_SETTINGS

logger = logging.getLogger(__name__)

class VocalPostProcessor:
    """Handles post-processing of separated vocals."""
    
    def __init__(self):
        self.noise_gate_threshold = -40  # dB
        self.attack_time = 10    # ms
        self.release_time = 50   # ms
    
    def process(self, audio_path: str) -> str:
        """Post-process vocals for better quality."""
        try:
            # Load audio
            audio = AudioSegment.from_wav(audio_path)
            
            # Apply processing chain
            audio = self._ensure_mono(audio)
            audio = self._normalize_loudness(audio)
            audio = self._apply_noise_gate(audio)
            
            # Export processed audio
            self._export_audio(audio, audio_path)
            
            return audio_path
            
        except Exception as e:
            logger.error(f"Post-processing failed: {str(e)}")
            raise AudioProcessingError(f"Vocal post-processing failed: {str(e)}")
    
    def _ensure_mono(self, audio: AudioSegment) -> AudioSegment:
        """Convert audio to mono if needed."""
        return audio.set_channels(1) if audio.channels > 1 else audio
    
    def _normalize_loudness(self, audio: AudioSegment) -> AudioSegment:
        """Normalize audio to target LUFS."""
        target_lufs = AUDIO_SETTINGS['target_lufs']
        current_lufs = audio.dBFS
        gain_change = target_lufs - current_lufs
        return audio.apply_gain(gain_change)
    
    def _apply_noise_gate(self, audio: AudioSegment) -> AudioSegment:
        """Apply noise gate to reduce background noise."""
        samples = np.array(audio.get_array_of_samples())
        window_size = int(audio.frame_rate * 0.02)  # 20ms windows
        
        # Calculate energy profile
        energy_profile = self._calculate_energy_profile(samples, window_size)
        
        # Apply gating
        gated_samples = self._gate_audio(
            samples, 
            energy_profile, 
            window_size
        )
        
        # Convert back to AudioSegment
        return audio._spawn(gated_samples)
    
    def _calculate_energy_profile(self, samples: np.ndarray, 
                                window_size: int) -> np.ndarray:
        """Calculate RMS energy profile of audio."""
        num_windows = len(samples) // window_size
        energy_profile = np.zeros(num_windows)
        
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window = samples[start:end]
            energy_profile[i] = np.sqrt(np.mean(window ** 2))
            
        return energy_profile
    
    def _gate_audio(self, samples: np.ndarray, energy_profile: np.ndarray,
                    window_size: int) -> np.ndarray:
        """Apply noise gate based on energy profile."""
        threshold = 10 ** (self.noise_gate_threshold / 20)
        attack_step = 1.0 / self.attack_time if self.attack_time > 0 else 1.0
        release_step = 1.0 / self.release_time if self.release_time > 0 else 1.0
        
        gated = samples.copy()
        current_gain = 1.0
        
        for i in range(len(energy_profile)):
            start = i * window_size
            end = start + window_size
            
            if energy_profile[i] < threshold:
                # Release
                current_gain = max(0.0, current_gain - release_step)
            else:
                # Attack
                current_gain = min(1.0, current_gain + attack_step)
                
            gated[start:end] = samples[start:end] * current_gain
            
        return gated
    
    def _export_audio(self, audio: AudioSegment, path: str):
        """Export processed audio with correct settings."""
        audio.export(
            path,
            format="wav",
            parameters=[
                "-ar", str(AUDIO_SETTINGS['sample_rate']),
                "-ac", str(AUDIO_SETTINGS['channels']),
                "-sample_fmt", "s16"
            ]
        )
