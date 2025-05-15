import os
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioFilePair:
    """Represents a matched pair of recv_out and trans_out files."""
    def __init__(self, 
                 base_key: str, 
                 recv_file: Optional[Path] = None, 
                 trans_file: Optional[Path] = None, 
                 timestamp: Optional[float] = None):
        self.base_key = base_key
        self.recv_file = recv_file
        self.trans_file = trans_file
        self.timestamp = timestamp or (
            recv_file.stat().st_mtime if recv_file else
            trans_file.stat().st_mtime if trans_file else
            datetime.now().timestamp()
        )
    
    @property
    def has_both_sides(self) -> bool:
        """Check if both sides of the conversation are present."""
        return self.recv_file is not None and self.trans_file is not None
    
    @property
    def formatted_timestamp(self) -> str:
        """Return timestamp as a formatted string for display."""
        dt = datetime.fromtimestamp(self.timestamp)
        return dt.strftime("%Y%m%d_%H%M%S")
    
    def __str__(self) -> str:
        return (f"AudioFilePair(base_key={self.base_key}, "
                f"recv={self.recv_file.name if self.recv_file else 'None'}, "
                f"trans={self.trans_file.name if self.trans_file else 'None'}, "
                f"timestamp={self.formatted_timestamp})")

def find_audio_pairs(input_dir: str) -> List[AudioFilePair]:
    """
    Find matching pairs of recv_out and trans_out files in the given directory.
    
    Args:
        input_dir: Directory containing audio files
        
    Returns:
        List of AudioFilePair objects representing matched file pairs
    """
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        logging.error(f"Input directory does not exist: {input_dir}")
        return []
    
    # Find all audio files in the directory (and subdirectories)
    audio_files = [
        f for f in input_path.glob("**/*.wav") 
        if f.is_file() and f.name.startswith(("recv_", "trans_"))
    ]
    
    logging.info(f"Found {len(audio_files)} audio files (recv_/trans_) in {input_dir}")
    
    # Group files by their base key (stripping recv_/trans_ prefix)
    pairs: Dict[str, AudioFilePair] = {}
    
    # Regular expression to extract the base key
    # Example: trans_out-12345678901-123-20250503-012345-1234567890.123.wav
    base_key_pattern = re.compile(r'(?:recv|trans)_(.+)\.wav$')
    
    for audio_file in audio_files:
        match = base_key_pattern.search(audio_file.name)
        if not match:
            logging.warning(f"Could not extract base key from: {audio_file.name}")
            continue
        
        base_key = match.group(1)
        
        # Create or update the pair
        if base_key not in pairs:
            pairs[base_key] = AudioFilePair(base_key)
        
        # Set the appropriate file based on prefix
        if audio_file.name.startswith("recv_"):
            pairs[base_key].recv_file = audio_file
        elif audio_file.name.startswith("trans_"):
            pairs[base_key].trans_file = audio_file
    
    # Convert dictionary to sorted list (by timestamp)
    result = sorted(pairs.values(), key=lambda p: p.timestamp)
    
    # Log statistics
    complete_pairs = sum(1 for p in result if p.has_both_sides)
    logging.info(f"Found {len(result)} unique conversations, {complete_pairs} with both sides")
    
    return result

def list_audio_pairs(pairs: List[AudioFilePair]) -> None:
    """Print information about matched audio pairs."""
    print(f"\nFound {len(pairs)} audio pair(s):")
    for i, pair in enumerate(pairs, 1):
        recv = pair.recv_file.name if pair.recv_file else "Missing"
        trans = pair.trans_file.name if pair.trans_file else "Missing"
        print(f"{i}. {pair.formatted_timestamp}: {recv} <-> {trans}")

if __name__ == "__main__":
    # Simple CLI for testing
    import argparse
    parser = argparse.ArgumentParser(description="Find matching recv_out/trans_out audio file pairs")
    parser.add_argument("input_dir", help="Directory containing audio files")
    args = parser.parse_args()
    
    pairs = find_audio_pairs(args.input_dir)
    list_audio_pairs(pairs) 