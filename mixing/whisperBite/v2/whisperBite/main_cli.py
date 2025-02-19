"""
Main CLI interface for WhisperBite.
"""

import os
import sys
import argparse
import logging
from typing import Optional

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whisperBite.config import ProcessingOptions, setup_logging, AudioProcessingError
from whisperBite.processor import AudioProcessor
from whisperBite.utils.file_handler import FileHandler

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="WhisperBite: Advanced audio processing and transcription tool"
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input_dir', type=str,
                            help='Directory containing input audio files')
    input_group.add_argument('--input_file', type=str,
                            help='Single audio file for processing')
    input_group.add_argument('--url', type=str,
                            help='URL to download audio from')
    
    # Output options
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save output files')
    
    # Processing options
    parser.add_argument('--model', type=str, default="turbo",  # Changed default to turbo
                       choices=["base", "small", "medium", "large", "turbo"],
                       help='Whisper model to use')
    parser.add_argument('--num_speakers', type=int, default=2,
                       help='Number of speakers for diarization')
    
    # Feature flags
    parser.add_argument('--word_level', action='store_true',
                       help='Enable word-level processing')
    parser.add_argument('--enable_diarization', action='store_true',
                       help='Enable speaker diarization')
    parser.add_argument('--enable_vocal_separation', action='store_true',
                       help='Enable vocal separation')
    
    # Additional options
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--keep_temp', action='store_true',
                       help='Keep temporary files')
    
    return parser.parse_args()

def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(verbose=args.verbose)
        
        # Create output directory
        output_dir = FileHandler.create_output_directory(args.output_dir)
        logging.info(f"Output directory: {output_dir}")
        
        # Create processing options
        options = ProcessingOptions(
            word_level=args.word_level,
            diarization=args.enable_diarization,
            vocal_separation=args.enable_vocal_separation,
            model_name=args.model,
            num_speakers=args.num_speakers,
            output_dir=output_dir
        )
        
        # Initialize processor
        processor = AudioProcessor(output_dir)
        
        # Process the audio
        if args.url:
            from whisperBite.utils.download import DownloadUtils
            input_path = DownloadUtils.download_audio(args.url, output_dir)
        elif args.input_file:
            input_path = args.input_file
        else:
            input_path = args.input_dir
            
        if not input_path or not os.path.exists(input_path):
            raise AudioProcessingError("No valid input provided")
            
        # Process the file(s)
        results = processor.process_audio(input_path, options)
        
        # Cleanup
        if not args.keep_temp:
            processor.cleanup()
        
        logging.info("Processing complete!")
        return 0
        
    except KeyboardInterrupt:
        logging.info("\nProcessing interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
