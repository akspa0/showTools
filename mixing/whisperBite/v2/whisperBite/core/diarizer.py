"""
Speaker diarization handler for WhisperBite.
Maps transcriptions to speaker segments and saves audio slices.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
from whisperBite.core.feature_extractor import AudioFeatures
from whisperBite.config import AudioProcessingError

logger = logging.getLogger(__name__)

class SpeakerDiarizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline: Optional[Pipeline] = None
        logger.info(f"Initializing SpeakerDiarizer with device: {self.device}")
        
    def process(self, audio_path: str, features: AudioFeatures, 
                num_speakers: Optional[int] = None) -> Dict:
        """Process audio for speaker diarization."""
        try:
            # Load source audio
            audio = AudioSegment.from_wav(audio_path)
            logger.info(f"Processing audio file: {audio_path}")
            
            # Initialize pipeline if needed
            if not self.pipeline:
                self._initialize_pipeline()
            
            # Get diarization
            diarization = self.pipeline(
                audio_path,
                num_speakers=num_speakers
            )
            
            # Process into speaker segments
            speaker_segments = {}
            current_segments = {}
            
            # Process each turn
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                
                # Convert times to milliseconds
                start_ms = int(turn.start * 1000)
                end_ms = int(turn.end * 1000)
                
                # Extract audio segment
                segment_audio = audio[start_ms:end_ms]
                
                # Create segment info
                segment = {
                    'start': start_ms,
                    'end': end_ms,
                    'duration': end_ms - start_ms,
                    'audio': segment_audio,
                    'text': None  # Will be filled with transcription
                }
                
                speaker_segments[speaker].append(segment)
            
            logger.info(f"Created {sum(len(segs) for segs in speaker_segments.values())} segments across {len(speaker_segments)} speakers")
            return speaker_segments
            
        except Exception as e:
            logger.error(f"Diarization failed: {str(e)}")
            raise AudioProcessingError(f"Diarization failed: {str(e)}")
    
    def _initialize_pipeline(self):
        """Initialize the pyannote pipeline."""
        try:
            token = os.getenv('HF_TOKEN')
            if not token:
                raise AudioProcessingError(
                    "HF_TOKEN environment variable not set. Required for pyannote.audio"
                )
            
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=token
            )
            
            # Move to appropriate device and set parameters
            self.pipeline = self.pipeline.to(self.device)
            
        except Exception as e:
            raise AudioProcessingError(f"Failed to initialize diarization pipeline: {str(e)}")
