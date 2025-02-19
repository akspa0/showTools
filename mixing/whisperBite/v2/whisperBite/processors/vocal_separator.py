"""
Main vocal separation interface for WhisperBite.
Coordinates the vocal separation process.
"""

import os
import logging
from whisperBite.config import AudioProcessingError
from .demucs_processor import DemucsProcessor
from .post_processor import VocalPostProcessor

logger = logging.getLogger(__name__)

class VocalSeparator:
    """Main interface for vocal separation."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.vocals_dir = os.path.join(output_dir, "vocals")
        os.makedirs(self.vocals_dir, exist_ok=True)
        
        # Initialize processors
        self.demucs = DemucsProcessor()
        self.post_processor = VocalPostProcessor()
    
    def separate(self, audio_path: str) -> str:
        """
        Separate vocals from audio.
        Returns path to isolated vocals.
        """
        try:
            logger.info(f"Starting vocal separation for: {audio_path}")
            
            # Verify input file exists
            if not os.path.exists(audio_path):
                raise AudioProcessingError(f"Input audio file not found: {audio_path}")
            
            # Prepare output path
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_path = os.path.join(self.vocals_dir, f"{base_name}_vocals.wav")
            logger.info(f"Will save vocals to: {output_path}")
            
            # Run Demucs separation
            logger.info("Running Demucs separation...")
            raw_vocals = self.demucs.process(audio_path, output_path)
            
            if not raw_vocals or not os.path.exists(raw_vocals):
                raise AudioProcessingError(f"Demucs failed to produce output at: {raw_vocals}")
            
            # Post-process vocals
            logger.info("Post-processing vocals...")
            processed_vocals = self.post_processor.process(raw_vocals)
            
            if not processed_vocals or not os.path.exists(processed_vocals):
                raise AudioProcessingError(f"Post-processing failed to produce output at: {processed_vocals}")
            
            logger.info(f"Vocal separation complete. Output at: {processed_vocals}")
            return processed_vocals
            
        except Exception as e:
            logger.error(f"Vocal separation failed: {str(e)}")
            raise AudioProcessingError(f"Vocal separation failed: {str(e)}")
