"""
Whisper transcription handler for WhisperBite.
Provides both structured JSON output and human-readable transcriptions.
"""

import os
import json
import logging
from typing import Dict, Optional
from datetime import datetime
import whisper
import torch
import numpy as np
from whisperBite.config import AudioProcessingError

logger = logging.getLogger(__name__)

class WhisperTranscriber:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        logger.info(f"Initializing WhisperTranscriber with device: {self.device}")
    
    def transcribe(self, audio_path: str, model_name: str = "base", output_dir: str = None) -> Dict:
        """
        Transcribe audio using Whisper and save both JSON and text outputs.
        Returns dictionary with transcription results.
        """
        try:
            model = self._get_model(model_name)
            
            # Get audio duration
            import ffmpeg
            probe = ffmpeg.probe(audio_path)
            duration = float(probe['streams'][0]['duration'])
            
            # Transcribe with all features enabled
            result = model.transcribe(
                audio_path,
                language=None,
                task="transcribe",
                word_timestamps=True,
                verbose=None
            )
            
            # Process results
            processed = self._process_transcription(result)
            
            # Add metadata
            metadata = {
                "file_info": {
                    "filename": os.path.basename(audio_path),
                    "duration": duration,
                    "processed_date": datetime.now().isoformat(),
                    "model": model_name
                },
                "transcription": processed
            }
            
            # Save outputs if directory provided
            if output_dir:
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                
                # Save JSON output
                json_path = os.path.join(output_dir, f"{base_name}_transcription.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                # Save human-readable transcription
                text_path = os.path.join(output_dir, f"{base_name}_transcription.txt")
                self._save_readable_transcription(metadata, text_path)
                
                logger.info(f"Saved transcription outputs to {output_dir}")
                metadata['output_files'] = {
                    'json': json_path,
                    'text': text_path
                }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise AudioProcessingError(f"Transcription failed: {str(e)}")
    
    def _process_transcription(self, result: Dict) -> Dict:
        """Process and structure transcription results."""
        processed = {
            'full_text': result.get('text', '').strip(),
            'language': result.get('language', 'unknown'),
            'segments': []
        }
        
        # Process segments with detailed information
        for segment in result.get('segments', []):
            processed_segment = {
                'index': len(processed['segments']),
                'start': float(segment['start']),
                'end': float(segment['end']),
                'text': segment['text'].strip(),
                'words': []
            }
            
            # Process word timestamps
            if 'words' in segment:
                for word in segment['words']:
                    word_text = word.get('word', word.get('text', '')).strip()
                    processed_segment['words'].append({
                        'text': word_text,
                        'start': float(word['start']),
                        'end': float(word['end']),
                        'confidence': float(word.get('probability', 0.0))
                    })
            
            if self._validate_segment(processed_segment):
                processed['segments'].append(processed_segment)
        
        # Add segment statistics
        processed['statistics'] = {
            'total_segments': len(processed['segments']),
            'total_words': sum(len(seg['words']) for seg in processed['segments']),
            'duration': processed['segments'][-1]['end'] if processed['segments'] else 0
        }
        
        return processed
    
    def _save_readable_transcription(self, metadata: Dict, output_path: str):
        """Save human-readable transcription with timestamps."""
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("WhisperBite Transcription\n")
            f.write("=======================\n\n")
            
            # Write file info
            file_info = metadata['file_info']
            f.write(f"File: {file_info['filename']}\n")
            f.write(f"Duration: {file_info['duration']:.2f} seconds\n")
            f.write(f"Processed: {file_info['processed_date']}\n")
            f.write(f"Model: {file_info['model']}\n\n")
            
            # Write full text
            f.write("Complete Transcription:\n")
            f.write("---------------------\n")
            f.write(metadata['transcription']['full_text'])
            f.write("\n\n")
            
            # Write segments with timestamps
            f.write("Segments:\n")
            f.write("---------\n")
            for segment in metadata['transcription']['segments']:
                timestamp = f"[{segment['start']:.2f} -> {segment['end']:.2f}]"
                f.write(f"{timestamp} {segment['text']}\n")
            
            # Write statistics
            stats = metadata['transcription']['statistics']
            f.write(f"\nStatistics:\n")
            f.write(f"Total segments: {stats['total_segments']}\n")
            f.write(f"Total words: {stats['total_words']}\n")
            f.write(f"Duration: {stats['duration']:.2f} seconds\n")
    
    def _validate_segment(self, segment: Dict) -> bool:
        """Validate a transcription segment."""
        if not all(key in segment for key in ['start', 'end', 'text']):
            return False
        if segment['end'] <= segment['start']:
            return False
        if not segment['text'].strip():
            return False
        return True
    
    def _get_model(self, model_name: str) -> whisper.Whisper:
        """Get cached model or load if not exists."""
        if model_name not in self.models:
            logger.info(f"Loading Whisper model: {model_name}")
            try:
                self.models[model_name] = whisper.load_model(model_name).to(self.device)
            except Exception as e:
                raise AudioProcessingError(f"Failed to load Whisper model: {str(e)}")
        return self.models[model_name]
    
    def cleanup(self):
        """Clean up loaded models."""
        logger.info("Cleaning up Whisper models")
        for model in self.models.values():
            del model
        self.models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
