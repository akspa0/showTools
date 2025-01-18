import os
import logging
import argparse
import re
from pathlib import Path
import numpy as np
from pydub import AudioSegment
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import shutil
import soundfile as sf
from scipy import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_audio(file_path):
    """Load audio file and convert to mono numpy array."""
    audio = AudioSegment.from_file(file_path)
    samples = np.array(audio.get_array_of_samples())
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels)).mean(axis=1)
    return samples, audio.frame_rate

def analyze_spectral_entropy(samples, sample_rate, frame_length=2048, hop_length=512):
    """Calculate spectral entropy to detect multiple voices."""
    # Compute spectrogram
    frequencies, times, spectrogram = signal.spectrogram(
        samples,
        fs=sample_rate,
        nperseg=frame_length,
        noverlap=frame_length - hop_length,
        window='hann'
    )
    
    # Normalize spectrogram
    spectrogram = spectrogram / spectrogram.sum(axis=0, keepdims=True)
    
    # Calculate entropy for each time frame
    entropy = -np.sum(spectrogram * np.log2(spectrogram + 1e-10), axis=0)
    
    # Higher entropy indicates more complex audio (potentially multiple voices)
    return np.mean(entropy)

def verify_single_speaker(file_path, diarization_pipeline, entropy_threshold=5.5, 
                         overlap_threshold=0.5, min_duration=0.2):
    """
    Verify if an audio file contains only a single speaker without overlapping voices.
    
    Args:
        file_path: Path to the audio file
        diarization_pipeline: Initialized pyannote diarization pipeline
        entropy_threshold: Threshold for spectral entropy (higher values indicate multiple voices)
        overlap_threshold: Maximum allowed overlap duration ratio
        min_duration: Minimum duration in seconds for valid segments
    
    Returns:
        tuple: (is_single_speaker, confidence_score)
    """
    try:
        # Load and analyze audio
        samples, sample_rate = load_audio(file_path)
        duration = len(samples) / sample_rate
        
        if duration < min_duration:
            return False, 0.0
        
        # Check spectral entropy
        entropy = analyze_spectral_entropy(samples, sample_rate)
        logging.info(f"{file_path}: Spectral entropy: {entropy:.2f} (threshold: {entropy_threshold})")
        if entropy > entropy_threshold:
            logging.info(f"{file_path}: Rejected - High spectral entropy ({entropy:.2f})")
            return False, 0.0
        
        # Run diarization
        with ProgressHook() as hook:
            diarization = diarization_pipeline(file_path, hook=hook)
        
        # Count unique speakers and check for overlaps
        speakers = set()
        total_speech = 0.0
        overlapping = 0.0
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.add(speaker)
            duration = turn.end - turn.start
            total_speech += duration
            
            # Check for overlapping speech
            for other_turn, _, other_speaker in diarization.itertracks(yield_label=True):
                if speaker != other_speaker:
                    overlap = min(turn.end, other_turn.end) - max(turn.start, other_turn.start)
                    if overlap > 0:
                        overlapping += overlap
        
        if len(speakers) > 1:
            logging.info(f"{file_path}: Rejected - Multiple speakers detected ({len(speakers)})")
            return False, 0.0
        
        overlap_ratio = overlapping / total_speech if total_speech > 0 else 1.0
        logging.info(f"{file_path}: Overlap ratio: {overlap_ratio:.2f} (threshold: {overlap_threshold})")
        if overlap_ratio > overlap_threshold:
            logging.info(f"{file_path}: Rejected - High overlap ratio ({overlap_ratio:.2f})")
            return False, 0.0
        
        # Calculate confidence score (0-1)
        entropy_score = max(0, 1 - (entropy / entropy_threshold))
        overlap_score = 1 - (overlap_ratio / overlap_threshold)
        confidence = (entropy_score + overlap_score) / 2  # Average instead of minimum
        
        logging.info(f"{file_path}: Confidence scores - Entropy: {entropy_score:.2f}, Overlap: {overlap_score:.2f}, Final: {confidence:.2f}")
        
        return True, confidence
        
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return False, 0.0

def is_speaker_directory(dirname):
    """Check if directory name matches the Speaker_SPEAKER_XX pattern."""
    pattern = r'^Speaker_SPEAKER_\d+$'
    return bool(re.match(pattern, dirname))

def sort_audio_files(input_dir, output_dir, confidence_threshold=0.5):
    """
    Sort audio files into clean (single speaker) and mixed categories.
    Maintains folder hierarchy for speaker directories.
    
    Args:
        input_dir: Directory containing speaker folders
        output_dir: Directory to store sorted files
        confidence_threshold: Minimum confidence score to consider a file clean
    """
    try:
        # Initialize pyannote pipeline
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
        pipeline.to(device)
        
        # Create base output directories
        clean_dir = os.path.join(output_dir, "clean")
        mixed_dir = os.path.join(output_dir, "mixed")
        os.makedirs(clean_dir, exist_ok=True)
        os.makedirs(mixed_dir, exist_ok=True)
        
        # Process all audio files
        total_files = 0
        clean_files = 0
        
        # Get all speaker directories
        for root, dirs, _ in os.walk(input_dir):
            # Filter for speaker directories
            speaker_dirs = [d for d in dirs if is_speaker_directory(d)]
            if not speaker_dirs:
                continue
                
            for speaker_dir in speaker_dirs:
                speaker_path = os.path.join(root, speaker_dir)
                # Create corresponding directories in clean and mixed outputs
                clean_speaker_dir = os.path.join(clean_dir, speaker_dir)
                mixed_speaker_dir = os.path.join(mixed_dir, speaker_dir)
                os.makedirs(clean_speaker_dir, exist_ok=True)
                os.makedirs(mixed_speaker_dir, exist_ok=True)
                
                # Process files in speaker directory
                for subroot, _, files in os.walk(speaker_path):
                    rel_path = os.path.relpath(subroot, speaker_path)
                    clean_subdir = os.path.join(clean_speaker_dir, rel_path)
                    mixed_subdir = os.path.join(mixed_speaker_dir, rel_path)
                    os.makedirs(clean_subdir, exist_ok=True)
                    os.makedirs(mixed_subdir, exist_ok=True)
                    
                    for file in files:
                        if file.endswith(('.wav', '.mp3', '.m4a')):
                            total_files += 1
                            file_path = os.path.join(subroot, file)
                            
                            logging.info(f"Processing: {file_path}")
                            is_clean, confidence = verify_single_speaker(file_path, pipeline)
                            
                            # Determine destination based on confidence score
                            if is_clean and confidence >= confidence_threshold:
                                dest_dir = clean_subdir
                                clean_files += 1
                            else:
                                dest_dir = mixed_subdir
                            
                            # Copy file to appropriate directory
                            dest_path = os.path.join(dest_dir, file)
                            shutil.copy2(file_path, dest_path)
                            logging.info(f"Moved to {dest_dir} with confidence {confidence:.2f}")
        
        if total_files == 0:
            logging.warning(f"No speaker directories found matching pattern 'Speaker_SPEAKER_XX' in {input_dir}")
        else:
            logging.info(f"Processing complete: {clean_files}/{total_files} files classified as clean")
        
        return clean_dir, mixed_dir
        
    except Exception as e:
        logging.error(f"Error during sorting: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Sort audio files based on speaker purity")
    parser.add_argument("input_dir", help="Directory containing input audio files")
    parser.add_argument("output_dir", help="Directory to store sorted files")
    parser.add_argument("--confidence", type=float, default=0.5,
                      help="Minimum confidence threshold (0-1) for clean files")
    args = parser.parse_args()
    
    try:
        clean_dir, mixed_dir = sort_audio_files(args.input_dir, args.output_dir, args.confidence)
        print(f"\nSorting complete!")
        print(f"Clean files: {clean_dir}")
        print(f"Mixed files: {mixed_dir}")
    except Exception as e:
        logging.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
