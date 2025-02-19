"""
Word-level audio processing for WhisperBite.
Handles splitting audio into word-level segments using Whisper's timestamps.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from pydub import AudioSegment
from whisperBite.core.feature_extractor import AudioFeatures
from whisperBite.config import AudioProcessingError

logger = logging.getLogger(__name__)

@dataclass
class WordSegment:
    """Container for word segment information."""
    text: str          # The word text
    start: float       # Start time in seconds
    end: float         # End time in seconds
    confidence: float  # Confidence score
    audio: Optional[AudioSegment] = None  # The audio segment
    index: int = 0     # Occurrence index of this word

class WordSplitter:
    """Handles word-level audio splitting and processing."""
    
    def __init__(self):
        self.min_word_length = 0.1    # Minimum word duration in seconds
        self.max_word_length = 2.0    # Maximum word duration in seconds
        self.min_confidence = 0.6     # Minimum confidence threshold
        self.phrase_gap = 0.5         # Maximum gap between words in a phrase
        self.min_phrase_words = 2     # Minimum words to form a phrase
    
    def process(self, audio_path: str, features: AudioFeatures, 
                transcription: Dict, output_dir: str, enable_phrases: bool = True) -> Dict:
        """
        Process audio into word and phrase segments using Whisper's timestamps.
        """
        try:
            # Load the full audio file
            audio = AudioSegment.from_wav(audio_path)
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            
            # First pass: Extract word segments from Whisper transcription
            word_segments = self._extract_word_segments(transcription, audio)
            
            # Second pass (optional): Group words into phrases
            if enable_phrases:
                phrase_segments = self._group_into_phrases(word_segments, audio)
            else:
                phrase_segments = []
            
            # Save all segments
            segment_files = self.save_segments(
                word_segments, 
                phrase_segments, 
                output_dir, 
                base_name
            )
            
            logger.info(f"Processed {len(word_segments)} words and {len(phrase_segments)} phrases")
            return {
                'word_segments': word_segments,
                'phrase_segments': phrase_segments,
                'files': segment_files
            }
            
        except Exception as e:
            logger.error(f"Word splitting failed: {str(e)}")
            raise AudioProcessingError(f"Word splitting failed: {str(e)}")
    
    def _extract_word_segments(self, transcription: Dict, 
                             audio: AudioSegment) -> List[WordSegment]:
        """Extract word segments using Whisper's timestamps."""
        word_segments = []
        word_counts = defaultdict(int)
        
        for segment in transcription.get('segments', []):
            # Check if segment has word-level info
            if 'words' not in segment:
                continue
                
            for word_data in segment['words']:
                # Validate word timing and confidence
                if not self._validate_word_data(word_data):
                    continue
                
                # Get word text and timing
                text = word_data['text'].strip().lower()
                if not text:  # Skip empty words
                    continue
                    
                start = float(word_data['start'])
                end = float(word_data['end'])
                
                # Extract audio for this word
                try:
                    word_audio = audio[int(start * 1000):int(end * 1000)]
                    word_counts[text] += 1
                    
                    word_segment = WordSegment(
                        text=text,
                        start=start,
                        end=end,
                        confidence=word_data.get('probability', 0.0),
                        audio=word_audio,
                        index=word_counts[text]
                    )
                    word_segments.append(word_segment)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract word segment '{text}': {str(e)}")
                    continue
        
        return sorted(word_segments, key=lambda x: x.start)
    
    def _group_into_phrases(self, word_segments: List[WordSegment], 
                           audio: AudioSegment) -> List[WordSegment]:
        """Group consecutive words into phrases based on timing."""
        phrases = []
        current_phrase = []
        phrase_counts = defaultdict(int)
        
        for i, segment in enumerate(word_segments):
            current_phrase.append(segment)
            
            # Check if we should end the current phrase
            end_phrase = False
            if i == len(word_segments) - 1:  # Last word
                end_phrase = True
            else:
                next_segment = word_segments[i + 1]
                gap = next_segment.start - segment.end
                if gap > self.phrase_gap:
                    end_phrase = True
            
            # Process completed phrase
            if end_phrase and len(current_phrase) >= self.min_phrase_words:
                phrase_text = " ".join(seg.text for seg in current_phrase)
                phrase_start = current_phrase[0].start
                phrase_end = current_phrase[-1].end
                
                # Extract audio for the whole phrase
                phrase_audio = audio[int(phrase_start * 1000):int(phrase_end * 1000)]
                phrase_counts[phrase_text] += 1
                
                phrase_segment = WordSegment(
                    text=phrase_text,
                    start=phrase_start,
                    end=phrase_end,
                    confidence=sum(seg.confidence for seg in current_phrase) / len(current_phrase),
                    audio=phrase_audio,
                    index=phrase_counts[phrase_text]
                )
                phrases.append(phrase_segment)
                current_phrase = []
        
        return sorted(phrases, key=lambda x: x.start)
    
    def _validate_word_data(self, word_data: Dict) -> bool:
        """Validate word data from transcription."""
        if not all(k in word_data for k in ['start', 'end', 'text']):
            return False
            
        if not word_data['text'].strip():
            return False
            
        duration = float(word_data['end']) - float(word_data['start'])
        if duration < self.min_word_length or duration > self.max_word_length:
            return False
            
        if word_data.get('probability', 1.0) < self.min_confidence:
            return False
            
        return True

    def save_segments(self, word_segments: List[WordSegment], 
                     phrase_segments: List[WordSegment],
                     output_dir: str, base_name: str) -> Dict[str, List[str]]:
        """Save word and phrase segments to organized folders."""
        segment_files = defaultdict(list)
        
        # Create base directories
        words_dir = os.path.join(output_dir, base_name, "words")
        phrases_dir = os.path.join(output_dir, base_name, "phrases")
        os.makedirs(words_dir, exist_ok=True)
        if phrase_segments:
            os.makedirs(phrases_dir, exist_ok=True)
        
        # Save words
        for segment in word_segments:
            clean_text = self._clean_text(segment.text)
            word_dir = os.path.join(words_dir, clean_text)
            os.makedirs(word_dir, exist_ok=True)
            
            # Save audio and info
            audio_path = os.path.join(word_dir, f"{clean_text}_{segment.index:03d}.wav")
            info_path = os.path.join(word_dir, f"{clean_text}_{segment.index:03d}.txt")
            
            segment.audio.export(audio_path, format="wav")
            segment_files['words'].append(audio_path)
            
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(self._create_info_text(segment))
        
        # Save phrases
        for segment in phrase_segments:
            clean_text = self._clean_text(segment.text)
            phrase_dir = os.path.join(phrases_dir, "sequence")
            os.makedirs(phrase_dir, exist_ok=True)
            
            # Save audio and info
            audio_path = os.path.join(phrase_dir, f"phrase_{len(segment_files['phrases']):03d}.wav")
            info_path = os.path.join(phrase_dir, f"phrase_{len(segment_files['phrases']):03d}.txt")
            
            segment.audio.export(audio_path, format="wav")
            segment_files['phrases'].append(audio_path)
            
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(self._create_info_text(segment))
        
        # Create summary files
        self._create_word_summary(words_dir, word_segments)
        if phrase_segments:
            self._create_phrase_summary(phrases_dir, phrase_segments)
        
        return dict(segment_files)

    def _clean_text(self, text: str) -> str:
        """Create clean, filesystem-safe text."""
        return ''.join(c for c in text.lower() if c.isalnum() or c == '_')

    def _create_info_text(self, segment: WordSegment) -> str:
        """Create info text for a segment."""
        return (
            f"Text: {segment.text}\n"
            f"Time: {segment.start:.3f} -> {segment.end:.3f}\n"
            f"Duration: {segment.end - segment.start:.3f}s\n"
            f"Confidence: {segment.confidence:.3f}\n"
        )

    def _create_word_summary(self, words_dir: str, segments: List[WordSegment]):
        """Create summary of word segments."""
        with open(os.path.join(words_dir, "word_summary.txt"), 'w') as f:
            f.write("Word Statistics\n==============\n\n")
            
            word_stats = defaultdict(list)
            for segment in segments:
                word_stats[segment.text].append(segment)
            
            for word, instances in sorted(word_stats.items()):
                f.write(f"{word}: {len(instances)} occurrences\n")
                total_duration = sum(seg.end - seg.start for seg in instances)
                f.write(f"  Total Duration: {total_duration:.2f}s\n")
                f.write(f"  Average Duration: {total_duration/len(instances):.2f}s\n")
                f.write("  Timestamps:\n")
                for seg in sorted(instances, key=lambda x: x.start):
                    f.write(f"    - {seg.start:.2f}s -> {seg.end:.2f}s\n")
                f.write("\n")

    def _create_phrase_summary(self, phrases_dir: str, segments: List[WordSegment]):
        """Create summary of phrase segments."""
        with open(os.path.join(phrases_dir, "phrase_summary.txt"), 'w') as f:
            f.write("Phrase Statistics\n================\n\n")
            for segment in sorted(segments, key=lambda x: x.start):
                f.write(f"Phrase: {segment.text}\n")
                f.write(f"Time: {segment.start:.2f}s -> {segment.end:.2f}s\n")
                f.write(f"Duration: {segment.end - segment.start:.2f}s\n\n")
