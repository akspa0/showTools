"""
Script to set up WhisperBite directory structure and create __init__.py files.
"""

import os

# Define the init file contents
INIT_FILES = {
    "whisperBite/__init__.py": '''"""
WhisperBite - Advanced audio processing and transcription tool.
"""

from .processor import AudioProcessor
from .config import ProcessingOptions, AudioProcessingError

__version__ = "1.0.0"
__all__ = ['AudioProcessor', 'ProcessingOptions', 'AudioProcessingError']
''',

    "whisperBite/processors/__init__.py": '''"""
WhisperBite processing modules.
"""

from .demucs_processor import DemucsProcessor
from .post_processor import VocalPostProcessor
from .vocal_separator import VocalSeparator

__all__ = ['DemucsProcessor', 'VocalPostProcessor', 'VocalSeparator']
''',

    "whisperBite/core/__init__.py": '''"""
WhisperBite core components.
"""

from .normalizer import AudioNormalizer
from .transcriber import WhisperTranscriber
from .diarizer import SpeakerDiarizer
from .word_splitter import WordSplitter
from .feature_extractor import FeatureExtractor

__all__ = [
    'AudioNormalizer',
    'WhisperTranscriber',
    'SpeakerDiarizer',
    'WordSplitter',
    'FeatureExtractor'
]
''',

    "whisperBite/utils/__init__.py": '''"""
WhisperBite utility modules.
"""

from .audio import AudioUtils
from .download import DownloadUtils
from .file_handler import FileHandler

__all__ = ['AudioUtils', 'DownloadUtils', 'FileHandler']
''',

    "tests/__init__.py": '# Tests for WhisperBite'
}

# Directory structure
DIRECTORIES = [
    "whisperBite",
    "whisperBite/processors",
    "whisperBite/core",
    "whisperBite/utils",
    "tests",
    "examples"
]

def create_directory_structure():
    """Create the directory structure for WhisperBite."""
    print("Creating directory structure...")
    
    # Create directories
    for directory in DIRECTORIES:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")

def create_init_files():
    """Create all __init__.py files with their content."""
    print("\nCreating __init__.py files...")
    
    for file_path, content in INIT_FILES.items():
        try:
            # Ensure directory exists
            directory = os.path.dirname(file_path)
            os.makedirs(directory, exist_ok=True)
            
            # Write file
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"Created: {file_path}")
        except Exception as e:
            print(f"Error creating {file_path}: {e}")

def main():
    """Main function to set up the project structure."""
    print("Setting up WhisperBite project structure...")
    
    # Create the basic directory structure
    create_directory_structure()
    
    # Create all __init__.py files
    create_init_files()
    
    print("\nSetup complete!")
    print("\nDirectory structure created:")
    for directory in DIRECTORIES:
        print(f"  {directory}/")
    
    print("\nNext steps:")
    print("1. Move existing source files to their appropriate directories")
    print("2. Create remaining module files")
    print("3. Set up tests and examples")

if __name__ == "__main__":
    main()
