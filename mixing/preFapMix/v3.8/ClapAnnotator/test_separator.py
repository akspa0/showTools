#!/usr/bin/env python
"""
Simple script to test if audio-separator is working correctly.
This doesn't require the CLAP model or other dependencies.
"""

import sys
import logging
import os
from pathlib import Path
from datetime import datetime

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

def ensure_dir(path):
    """Ensure a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)
    return path

def main():
    """Test audio-separator functionality."""
    # Check if input file is provided
    if len(sys.argv) < 2:
        print("Usage: python test_separator.py <audio_file>")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    if not input_file.is_file():
        log.error(f"Input file not found: {input_file}")
        sys.exit(1)
    
    log.info(f"Testing audio separation on: {input_file}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("ClapAnnotator_Output") / f"{input_file.stem}_{timestamp}"
    ensure_dir(output_dir)
    log.info(f"Output will be saved to: {output_dir}")
    
    try:
        # Import the Separator class
        from audio_separator.separator import Separator
        log.info("Successfully imported audio_separator.separator.Separator")
        
        # Create a separator instance with output directory specified
        separator = Separator(output_dir=str(output_dir))
        log.info("Successfully created Separator instance")
        
        # Print separator methods to understand the API
        log.info(f"Available methods: {dir(separator)}")
        
        # List supported models
        supported_models = separator.list_supported_model_files()
        log.info(f"Supported models: {supported_models}")
        
        # Load the specified model
        model_name = "mel_band_roformer_vocals_fv4_gabox.ckpt"
        model_name_base = model_name.replace('.ckpt', '')  # Remove .ckpt extension for output filename matching
        log.info(f"Loading model: {model_name}")
        separator.load_model(model_name)
        log.info(f"Model {model_name} loaded successfully")
        
        # Separate the audio file - no output_names parameter
        log.info(f"Separating audio file: {input_file}")
        output_files = separator.separate(str(input_file))
        
        # Print the output files
        log.info("Separation completed successfully!")
        log.info(f"Output files: {output_files}")
        
        # Print full paths to output files
        for output_file in output_files:
            full_path = output_dir / Path(output_file).name
            log.info(f"Full path: {full_path}")
            
        # Check if expected output files were generated
        expected_vocals = f"{input_file.stem}_(Vocals)_{model_name_base}.wav"
        expected_instrumental = f"{input_file.stem}_(Instrumental)_{model_name_base}.wav"
        
        log.info(f"Expected vocals file: {expected_vocals}")
        log.info(f"Expected instrumental file: {expected_instrumental}")
        
        output_files_set = {Path(f).name for f in output_files}
        if expected_vocals in output_files_set:
            log.info(f"Found expected vocals file: {expected_vocals}")
        else:
            log.warning(f"Could not find expected vocals file: {expected_vocals}")
            
        if expected_instrumental in output_files_set:
            log.info(f"Found expected instrumental file: {expected_instrumental}")
        else:
            log.warning(f"Could not find expected instrumental file: {expected_instrumental}")
        
    except ImportError as e:
        log.error(f"Failed to import audio_separator: {e}")
        log.error("Make sure audio-separator is installed correctly.")
        sys.exit(1)
    except Exception as e:
        log.error(f"Error during separation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 