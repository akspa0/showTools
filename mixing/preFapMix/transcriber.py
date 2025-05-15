import os
import logging
import subprocess
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WhisperTranscriber:
    """
    Interface to OpenAI Whisper for audio transcription.
    
    This class provides methods to transcribe audio files using the Whisper model,
    with support for different model sizes and output formats.
    """
    
    # Available Whisper models and their approximate VRAM requirements
    MODELS = {
        "tiny": {"vram": 1, "description": "Fastest, lowest accuracy"},
        "base": {"vram": 1, "description": "Fast, low accuracy"},
        "small": {"vram": 2, "description": "Balanced speed/accuracy"},
        "medium": {"vram": 5, "description": "Slower, good accuracy"},
        "large": {"vram": 10, "description": "Slowest, best accuracy"},
        "large-v2": {"vram": 10, "description": "Improved large model"},
        "large-v3": {"vram": 10, "description": "Latest and most accurate model"}
    }
    
    def __init__(self, 
                 model_name: str = "large-v3",
                 output_dir: Optional[str] = None,
                 device: str = "auto",
                 language: str = "en"):
        """
        Initialize the transcriber.
        
        Args:
            model_name: Whisper model to use (default: "large-v3")
            output_dir: Directory to save transcriptions
                (default: creates a temporary directory)
            device: Device to use for inference ("cpu", "cuda", or "auto")
            language: Language code for transcription (default: "en")
        """
        if model_name not in self.MODELS:
            logging.warning(f"Unknown model: {model_name}. Using large-v3 instead.")
            model_name = "large-v3"
            
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = device
        self.language = language
        
        # Check if whisper is installed
        try:
            self._run_command(["whisper", "--help"], "Checking Whisper")
        except Exception as e:
            logging.error(f"OpenAI Whisper not found. Please install it with 'pip install openai-whisper'")
            logging.error(f"Error: {e}")
    
    def _run_command(self, command: List[str], description: str) -> subprocess.CompletedProcess:
        """Run a command with proper logging."""
        logging.debug(f"Running command: {' '.join(command)}")
        try:
            result = subprocess.run(
                command, 
                check=True, 
                capture_output=True, 
                text=True
            )
            logging.debug(f"{description} completed successfully.")
            return result
        except subprocess.CalledProcessError as e:
            logging.error(f"{description} failed with error: {e}")
            logging.error(f"Error output: {e.stderr}")
            raise
    
    def transcribe(self, 
                  input_path: str, 
                  output_dir: Optional[str] = None,
                  output_formats: List[str] = ["txt", "srt", "json"],
                  speaker_detection: bool = False,
                  verbose: bool = False) -> Dict[str, str]:
        """
        Transcribe audio file using Whisper.
        
        Args:
            input_path: Path to input audio file
            output_dir: Directory to save output (overrides constructor setting)
            output_formats: List of output formats to generate
            speaker_detection: Whether to attempt speaker detection
            verbose: Whether to enable verbose logging
            
        Returns:
            Dictionary mapping format types to output file paths
        """
        input_file = Path(input_path)
        if not input_file.exists():
            logging.error(f"Input file does not exist: {input_path}")
            return {}
        
        # Determine output directory
        target_output_dir = output_dir or self.output_dir
        if not target_output_dir:
            target_output_dir = tempfile.mkdtemp()
            logging.info(f"Created temporary output directory: {target_output_dir}")
        else:
            os.makedirs(target_output_dir, exist_ok=True)
        
        # Build command
        command = [
            "whisper", input_path,
            "--model", self.model_name,
            "--output_dir", target_output_dir,
            "--language", self.language
        ]
        
        # Add device specification
        if self.device != "auto":
            command.extend(["--device", self.device])
        
        # Add output formats
        if output_formats:
            command.append("--output_format")
            command.append(",".join(output_formats))
        
        # Add verbose flag if specified
        if verbose:
            command.append("--verbose")
            
        # Add speaker detection if specified
        if speaker_detection:
            command.extend(["--detect_speakers", "True"])
        
        # Run the transcription command
        try:
            self._run_command(command, "Audio transcription")
            
            # Collect output files
            base_name = input_file.stem
            result = {}
            
            for format_type in output_formats:
                expected_path = os.path.join(target_output_dir, f"{base_name}.{format_type}")
                if os.path.exists(expected_path):
                    result[format_type] = expected_path
            
            logging.info(f"Transcription completed. Found outputs: {list(result.keys())}")
            return result
            
        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            return {}
    
    def get_transcript_text(self, transcript_path: str) -> str:
        """
        Get the transcript text from a transcript file.
        
        Args:
            transcript_path: Path to transcript file (.txt or .json)
            
        Returns:
            Transcript text as a string
        """
        if not os.path.exists(transcript_path):
            logging.error(f"Transcript file does not exist: {transcript_path}")
            return ""
        
        try:
            ext = os.path.splitext(transcript_path)[1].lower()
            
            if ext == '.txt':
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            elif ext == '.json':
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Extract text from segments
                    if 'segments' in data:
                        return " ".join(segment.get('text', '') for segment in data['segments'])
                    # Fallback to text field
                    elif 'text' in data:
                        return data['text']
                    else:
                        logging.warning(f"Could not find transcript text in JSON: {transcript_path}")
                        return ""
            
            else:
                logging.warning(f"Unsupported transcript format: {ext}")
                return ""
                
        except Exception as e:
            logging.error(f"Error reading transcript: {e}")
            return ""

if __name__ == "__main__":
    # Simple CLI for testing
    import argparse
    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper")
    parser.add_argument("input_file", help="Input audio file")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--model", default="large-v3", help="Whisper model size")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to use")
    parser.add_argument("--language", default="en", help="Language code")
    parser.add_argument("--formats", default="txt,srt,json", help="Output formats (comma-separated)")
    parser.add_argument("--speakers", action="store_true", help="Enable speaker detection")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    formats = args.formats.split(',')
    
    transcriber = WhisperTranscriber(
        model_name=args.model,
        output_dir=args.output_dir,
        device=args.device,
        language=args.language
    )
    
    output_files = transcriber.transcribe(
        args.input_file,
        output_formats=formats,
        speaker_detection=args.speakers,
        verbose=args.verbose
    )
    
    print("\nTranscription completed:")
    for format_type, path in output_files.items():
        print(f"  - {format_type}: {path}")
        
    # Print the text transcript if available
    if 'txt' in output_files:
        print("\nTranscript:")
        print(transcriber.get_transcript_text(output_files['txt'])) 