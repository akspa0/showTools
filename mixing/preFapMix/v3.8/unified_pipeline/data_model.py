from pathlib import Path
from typing import List, Dict, Optional

class AudioFile:
    """Represents an audio file and its associated metadata and processing outputs."""
    def __init__(self, path: Path):
        self.path = path
        self.metadata: Dict = {}
        self.separated_stems: Dict[str, Path] = {}
        self.annotations: Optional[Dict] = None

class Call:
    """Represents a processed call, including all outputs and metadata."""
    def __init__(self, input_audio: AudioFile):
        self.input_audio = input_audio
        self.separated_stems: Dict[str, Path] = {}
        self.clap_annotations: Optional[Dict] = None
        self.mixed_outputs: Dict[str, Path] = {}
        self.transcripts: Dict[str, Path] = {}
        self.diarization: Optional[Dict] = None
        self.soundbites: List[Path] = []
        self.metadata: Dict = {}

class Show:
    """Represents a collection of calls and show-level metadata/output."""
    def __init__(self, name: str):
        self.name = name
        self.calls: List[Call] = []
        self.metadata: Dict = {}
        self.final_show_file: Optional[Path] = None 