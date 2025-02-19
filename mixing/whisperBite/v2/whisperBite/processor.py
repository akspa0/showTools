"""
Main processing pipeline for WhisperBite.
"""

import os
import sys
import logging
from typing import Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whisperBite.config import ProcessingOptions, AudioProcessingError, CACHE_SETTINGS
from whisperBite.core.normalizer import AudioNormalizer
from whisperBite.core.feature_extractor import FeatureExtractor
from whisperBite.core.transcriber import WhisperTranscriber
from whisperBite.core.diarizer import SpeakerDiarizer
from whisperBite.core.word_splitter import WordSplitter
from whisperBite.processors.vocal_separator import VocalSeparator
from whisperBite.utils.file_handler import FileHandler

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.cache = {}
        self.temp_files = []
        
        # Initialize components
        self.normalizer = AudioNormalizer()
        self.feature_extractor = FeatureExtractor()
        self.transcriber = WhisperTranscriber()
        self.diarizer = SpeakerDiarizer()
        self.word_splitter = WordSplitter()
        self.vocal_separator = VocalSeparator(output_dir)

    def process_audio(self, input_path: str, options: ProcessingOptions) -> Dict:
        """Process audio and create archive of results."""
        try:
            results = {}
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            
            # 1. Initial Normalization
            logger.info("Normalizing audio...")
            normalized_path = self._get_or_create_normalized(input_path)
            self.temp_files.append(normalized_path)
            
            # 2. Vocal Separation (if enabled) - Moved up in the pipeline
            processing_path = normalized_path  # Path to use for further processing
            if options.vocal_separation:
                logger.info("Performing vocal separation...")
                vocals_path = self.vocal_separator.separate(normalized_path)
                if vocals_path:
                    # Save vocals in output structure
                    vocal_output = FileHandler.save_vocals(
                        vocals_path,
                        self.output_dir,
                        base_name
                    )
                    self.temp_files.append(vocals_path)
                    results['vocals_path'] = vocal_output
                    # Use the separated vocals for further processing
                    processing_path = vocal_output
            
            # 3. Feature Extraction on final audio
            logger.info("Extracting audio features...")
            features = self.feature_extractor.extract_features(processing_path)
            
            # 4. Transcription
            logger.info("Performing transcription...")
            transcription = self.transcriber.transcribe(
                processing_path,
                model_name=options.model_name
            )
            results['transcription'] = transcription
            
            # 5. Diarization (if enabled)
            if options.diarization:
                logger.info("Performing speaker diarization...")
                speaker_segments = self.diarizer.process(
                    processing_path,
                    features,
                    num_speakers=options.num_speakers
                )
                
                # Save diarization results
                logger.info("Saving speaker segments...")
                diar_results = FileHandler.save_diarization_results(
                    speaker_segments,
                    self.output_dir,
                    input_path,  # Keep original name for output structure
                    transcription
                )
                self.temp_files.extend(diar_results.get('temp_files', []))
                results['speaker_segments'] = speaker_segments
            
            # 6. Word-level processing (if enabled)
            if options.word_level:
                logger.info("Performing word-level processing...")
                word_segments = self.word_splitter.process(
                    processing_path,
                    features,
                    transcription,
                    self.output_dir
                )

            # Create archive of results
            archive_path = FileHandler.create_archive(self.output_dir, input_path)
            results['archive_path'] = archive_path
            
            return results
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            self.cleanup()
            raise AudioProcessingError(f"Processing pipeline failed: {str(e)}")

    def _get_or_create_normalized(self, input_path: str) -> str:
        """Get cached normalized audio or create if not exists."""
        cache_key = f"norm_{input_path}"
        
        if cache_key not in self.cache:
            logger.info(f"Normalizing audio: {input_path}")
            normalized_path = self.normalizer.normalize(input_path)
            self.cache[cache_key] = normalized_path
            self._cleanup_cache_if_needed()
            
        return self.cache[cache_key]

    def _cleanup_cache_if_needed(self):
        """Clean up cache if it exceeds size limit."""
        cache_size = sum(
            os.path.getsize(path) 
            for path in self.cache.values() 
            if os.path.exists(path)
        )
        
        if cache_size > CACHE_SETTINGS['max_size']:
            logger.info("Cleaning up cache")
            while cache_size > CACHE_SETTINGS['max_size'] * CACHE_SETTINGS['cleanup_threshold']:
                oldest_key = min(self.cache.keys())
                path = self.cache.pop(oldest_key)
                if os.path.exists(path):
                    os.remove(path)
                    cache_size -= os.path.getsize(path)

    def cleanup(self):
        """Clean up all temporary files."""
        logger.info("Cleaning up temporary files")
        FileHandler.cleanup_temporary_files(self.temp_files)
        self.temp_files.clear()
        
        for path in self.cache.values():
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                logger.error(f"Cleanup error: {str(e)}")
        self.cache.clear()
