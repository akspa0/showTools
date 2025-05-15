#!/usr/bin/env python
import argparse
import logging
import sys
import json
from pathlib import Path
import traceback
import os
import shutil

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# Define paths manually to avoid importing settings
PROJECT_ROOT = Path(__file__).parent.resolve()
TEMP_OUTPUT_DIR = PROJECT_ROOT / 'temp_output'
BASE_OUTPUT_DIR = PROJECT_ROOT / 'ClapAnnotator_Output'
AUDIO_SEPARATOR_MODEL_DIR = PROJECT_ROOT / '_models' / 'audio-separator'

# Ensure directories exist
def ensure_dir(path):
    """Ensure a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)
    return path

# Clean up directory
def cleanup_directory(directory):
    """Remove all files in a directory but keep the directory itself."""
    if not directory.exists():
        return
    
    for item in directory.iterdir():
        if item.is_file():
            try:
                item.unlink()
            except Exception as e:
                log.error(f"Failed to delete {item}: {e}")
        elif item.is_dir():
            try:
                shutil.rmtree(item)
            except Exception as e:
                log.error(f"Failed to delete directory {item}: {e}")

def check_ffmpeg():
    """Check if ffmpeg is available in the system."""
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False

def process_audio_file(
    input_file: Path,
    output_dir: Path = None,
    cleanup_temp: bool = True
):
    """
    Process a single audio file - simplified version that only checks the file exists
    and creates output directories.
    """
    if not input_file.is_file():
        log.error(f"Input file not found: {input_file}")
        return {"success": False, "error": "Input file not found"}
    
    # Create output directory with timestamp
    if output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = BASE_OUTPUT_DIR / f"{input_file.stem}_{timestamp}"
    
    ensure_dir(output_dir)
    ensure_dir(TEMP_OUTPUT_DIR)
    
    log.info(f"Processing file: {input_file}")
    log.info(f"Output will be saved to: {output_dir}")
    
    try:
        # Just copy the file to the output directory as a test
        output_file = output_dir / input_file.name
        shutil.copy2(input_file, output_file)
        
        log.info(f"File copied to: {output_file}")
        
        # Create a simple JSON result
        result = {
            "input_file": str(input_file),
            "output_file": str(output_file),
            "message": "This is a simplified CLI that only copies the file. Full functionality requires fixing dependency issues."
        }
        
        # Save the result
        result_file = output_dir / f"{input_file.stem}_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        return {
            "success": True,
            "output_file": str(output_file),
            "result_file": str(result_file)
        }
    
    except Exception as e:
        log.exception(f"Error processing {input_file}: {e}")
        return {"success": False, "error": str(e)}
    
    finally:
        if cleanup_temp:
            log.info("Cleaning up temporary files...")
            cleanup_directory(TEMP_OUTPUT_DIR)

def check_environment():
    """Check if the environment is properly configured for the simplified CLI."""
    issues = []
    
    # Check Python version
    import platform
    python_version = platform.python_version()
    log.info(f"Python version: {python_version}")
    
    # Check for ffmpeg
    if not check_ffmpeg():
        issues.append("ffmpeg not found in PATH. Please install ffmpeg and ensure it's in your PATH")
    else:
        log.info("ffmpeg is available")
    
    # Check for .env file
    if not (PROJECT_ROOT / '.env').is_file():
        issues.append("No .env file found. Please create one with your HF_TOKEN")
    
    return issues

def main():
    """Main CLI entry point for the simplified version."""
    parser = argparse.ArgumentParser(
        description="Simplified CLAP Annotator CLI - For testing basic functionality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add a check-environment command
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check if the environment is properly configured"
    )
    
    # Input options
    parser.add_argument(
        "input", 
        type=str, 
        nargs="?",
        help="Path to input audio file"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir", 
        type=str, 
        help="Custom output directory (default: auto-generated based on input file)"
    )
    
    parser.add_argument(
        "--keep-temp", 
        action="store_true", 
        help="Keep temporary files after processing"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check environment if requested
    if args.check_env:
        log.info("Checking environment...")
        issues = check_environment()
        if issues:
            log.warning("Environment check found issues:")
            for i, issue in enumerate(issues, 1):
                log.warning(f"  {i}. {issue}")
            log.info("Please fix these issues before using the full functionality.")
        else:
            log.info("Basic environment check passed!")
        return
    
    # Check if input is provided
    if args.input is None:
        parser.print_help()
        sys.exit(1)
    
    # Process input
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    if input_path.is_file():
        result = process_audio_file(
            input_file=input_path,
            output_dir=output_dir,
            cleanup_temp=not args.keep_temp
        )
        
        if result.get("success", False):
            log.info("Processing completed successfully")
            log.info(f"File copied to: {result.get('output_file')}")
            log.info(f"Result saved to: {result.get('result_file')}")
        else:
            log.error(f"Processing failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    else:
        log.error(f"Input path does not exist or is not a file: {input_path}")
        sys.exit(1)

if __name__ == "__main__":
    main() 