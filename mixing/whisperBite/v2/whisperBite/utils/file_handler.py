"""
File handling utilities for WhisperBite.
Handles saving and organization of all output files.
"""

import os
import shutil
import zipfile
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pydub import AudioSegment
from whisperBite.config import AudioProcessingError

logger = logging.getLogger(__name__)

class FileHandler:
    @staticmethod
    def create_output_directory(base_dir: str) -> str:
        """Create timestamped output directory."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(base_dir, f"output_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
        return output_dir

    @staticmethod
    def _create_safe_filename(text: str, max_length: int = 100, prefix: Optional[int] = None) -> str:
        """
        Create filesystem-safe filename from text.
        Args:
            text: The text to convert to a filename
            max_length: Maximum length of the resulting filename
            prefix: Optional numeric prefix for the filename
        """
        # Remove invalid characters and excess whitespace
        safe_text = "".join(
            char if char.isalnum() or char in "- " else "_"
            for char in text.strip()
        ).strip()
        
        # Remove multiple consecutive underscores or spaces
        while "__" in safe_text:
            safe_text = safe_text.replace("__", "_")
        while "  " in safe_text:
            safe_text = safe_text.replace("  ", " ")
        
        # Add prefix if provided
        if prefix is not None:
            prefix_str = f"{prefix:03d}_"
            max_length = max_length - len(prefix_str)
        else:
            prefix_str = ""
        
        # Truncate to max length while keeping words intact
        if len(safe_text) > max_length:
            words = safe_text[:max_length].split()
            safe_text = "_".join(words[:-1])  # Drop the last partial word
            
            # Ensure we don't end with an underscore
            safe_text = safe_text.rstrip("_")
        
        final_name = prefix_str + safe_text
        return final_name or "unnamed_segment"

    @staticmethod
    def save_diarization_results(speaker_segments: Dict, 
                               base_output_dir: str,
                               original_audio_name: str,
                               transcription_results: Dict) -> Dict:
        """
        Save speaker segments with original directory structure.
        Returns mapping of temporary files for cleanup.
        """
        temp_files = []
        try:
            # Create base directory for this audio file
            base_name = os.path.splitext(os.path.basename(original_audio_name))[0]
            result_dir = os.path.join(base_output_dir, base_name)
            
            # Create required directories
            speaker_dir = os.path.join(result_dir, "speakers")
            os.makedirs(speaker_dir, exist_ok=True)
            
            # Track used filenames to avoid collisions
            used_filenames = set()
            
            # Process each speaker's segments
            for speaker, segments in speaker_segments.items():
                # Create speaker directories
                speaker_audio_dir = os.path.join(speaker_dir, f"Speaker_{speaker}")
                speaker_trans_dir = os.path.join(result_dir, f"Speaker_{speaker}_transcriptions")
                os.makedirs(speaker_audio_dir, exist_ok=True)
                os.makedirs(speaker_trans_dir, exist_ok=True)
                
                # Process each segment
                for idx, segment in enumerate(segments):
                    # Get corresponding transcription if available
                    segment_text = FileHandler._find_segment_transcription(
                        segment, transcription_results
                    )
                    
                    if segment_text:
                        # Create filename from transcription with numeric prefix
                        safe_name = FileHandler._create_safe_filename(
                            segment_text, 
                            max_length=120,  # Leave room for extensions and paths
                            prefix=idx + 1  # 1-based indexing for display
                        )
                        
                        # Handle filename collisions
                        base_safe_name = safe_name
                        counter = 1
                        while safe_name in used_filenames:
                            safe_name = f"{base_safe_name}_{counter}"
                            counter += 1
                        used_filenames.add(safe_name)
                        
                        audio_filename = f"{safe_name}.wav"
                        trans_filename = f"{safe_name}.txt"
                        
                        # Save audio segment
                        audio_path = os.path.join(speaker_audio_dir, audio_filename)
                        segment['audio'].export(audio_path, format="wav")
                        
                        # Save transcription with timing info
                        trans_path = os.path.join(speaker_trans_dir, trans_filename)
                        with open(trans_path, 'w', encoding='utf-8') as f:
                            f.write(f"[{segment['start']/1000:.2f} -> {segment['end']/1000:.2f}]\n")
                            f.write(segment_text + "\n")
                        
                        # Update segment with paths
                        segment['audio_path'] = audio_path
                        segment['text'] = segment_text
                    else:
                        # Fallback naming if no transcription
                        safe_name = FileHandler._create_safe_filename(
                            f"segment_{idx + 1}",
                            max_length=120
                        )
                        audio_path = os.path.join(speaker_audio_dir, f"{safe_name}.wav")
                        segment['audio'].export(audio_path, format="wav")
                        segment['audio_path'] = audio_path
            
            return {'temp_files': temp_files}
            
        except Exception as e:
            logger.error(f"Error saving diarization results: {str(e)}")
            raise AudioProcessingError(f"Failed to save diarization results: {str(e)}")

    @staticmethod
    def save_vocals(vocals_path: str, output_dir: str, base_name: str) -> str:
        """Save vocal separation results."""
        try:
            vocals_dir = os.path.join(output_dir, base_name, "demucs")
            os.makedirs(vocals_dir, exist_ok=True)
            
            # Copy vocals to output directory
            output_path = os.path.join(vocals_dir, "vocals.wav")
            shutil.copy2(vocals_path, output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving vocals: {str(e)}")
            raise AudioProcessingError(f"Failed to save vocals: {str(e)}")

    @staticmethod
    def _find_segment_transcription(segment: Dict, transcription_results: Dict) -> str:
        """Find corresponding transcription for a segment."""
        if not transcription_results or 'segments' not in transcription_results:
            return None
            
        # Convert segment times to seconds for comparison
        segment_start = segment['start'] / 1000  # ms to seconds
        segment_end = segment['end'] / 1000
        
        # Find overlapping transcription segments
        matching_text = []
        for trans_segment in transcription_results['segments']:
            # Check for overlap
            if (trans_segment['end'] > segment_start and 
                trans_segment['start'] < segment_end):
                matching_text.append(trans_segment['text'].strip())
        
        return " ".join(matching_text) if matching_text else None

    @staticmethod
    def create_archive(output_dir: str, original_filename: str) -> str:
        """Create a zip archive of only the essential output files."""
        try:
            # Create archive name
            base_name = os.path.splitext(os.path.basename(original_filename))[0]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_name = f"{base_name}_whisperBite_{timestamp}.zip"
            archive_path = os.path.join(output_dir, archive_name)
            
            # Create the zip file
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(output_dir):
                    # Skip certain directories
                    if any(x in root for x in ['normalized', 'demucs', 'vocals']):
                        continue
                        
                    for file in files:
                        # Skip the archive itself and any temporary files
                        if file == archive_name or file.endswith(('.tmp', '.temp')):
                            continue
                            
                        file_path = os.path.join(root, file)
                        # Only include wav files from speakers/ and words/ directories
                        # and txt files from *_transcriptions/ directories
                        if (('speakers' in root or 'words' in root) and file.endswith('.wav')) or \
                           ('transcriptions' in root and file.endswith('.txt')):
                            # Get relative path from output directory
                            rel_path = os.path.relpath(file_path, output_dir)
                            zipf.write(file_path, rel_path)
                
                # Add README
                readme_content = (
                    "WhisperBite Processing Results\n"
                    "============================\n\n"
                    f"Original File: {original_filename}\n"
                    f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    "Directory Structure:\n"
                    "- speakers/          : Speaker-separated audio segments\n"
                    "- words/            : Individual word segments\n"
                    "- *_transcriptions/ : Text transcriptions\n\n"
                    "Note: Files are numbered by their sequence in the original audio.\n"
                )
                zipf.writestr("README.txt", readme_content)
            
            logger.info(f"Created archive: {archive_path}")
            return archive_path
            
        except Exception as e:
            logger.error(f"Failed to create archive: {str(e)}")
            if 'archive_path' in locals() and os.path.exists(archive_path):
                try:
                    os.remove(archive_path)
                except:
                    pass
            raise AudioProcessingError(f"Failed to create archive: {str(e)}")

    @staticmethod
    def cleanup_temporary_files(temp_files: List[str]):
        """Clean up temporary files."""
        for file_path in temp_files:
            try:
                if os.path.exists(file_path):
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup {file_path}: {str(e)}")
