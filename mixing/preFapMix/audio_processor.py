import os
import re
import logging
import subprocess
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioProcessor:
    """
    Audio processing using FFmpeg for normalization, resampling, and mixing.
    
    This class provides methods to process audio files using FFmpeg, including:
    - Normalizing audio to a target LUFS level
    - Resampling audio to a target sample rate
    - Mixing left and right channels with adjustable volumes
    - Adding tones to the end of audio files
    """
    
    def __init__(self, 
                 target_sample_rate: int = 44100,
                 target_lufs: float = -14.0):
        """
        Initialize the audio processor.
        
        Args:
            target_sample_rate: Target sample rate in Hz (default: 44100)
            target_lufs: Target loudness level in LUFS (default: -14.0)
        """
        self.target_sample_rate = target_sample_rate
        self.target_lufs = target_lufs
        
        # Check if ffmpeg is installed
        try:
            self._run_command(["ffmpeg", "-version"], "Checking FFmpeg")
        except Exception as e:
            logging.error(f"FFmpeg not found. Please ensure it is installed and in your PATH.")
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
    
    def get_audio_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get audio file information using FFmpeg.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with audio information
        """
        if not os.path.exists(file_path):
            logging.error(f"File does not exist: {file_path}")
            return {}
        
        try:
            command = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", file_path
            ]
            
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            data = json.loads(result.stdout)
            
            # Extract relevant audio information
            audio_info = {}
            
            # Get audio stream info
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "audio":
                    audio_info["codec"] = stream.get("codec_name")
                    audio_info["sample_rate"] = int(stream.get("sample_rate", 0))
                    audio_info["channels"] = stream.get("channels", 0)
                    audio_info["bit_depth"] = stream.get("bits_per_sample")
                    audio_info["duration"] = float(stream.get("duration", 0))
                    break
            
            # Get format info
            if "format" in data:
                audio_info["format"] = data["format"].get("format_name")
                audio_info["duration"] = float(data["format"].get("duration", audio_info.get("duration", 0)))
                audio_info["size"] = int(data["format"].get("size", 0))
                audio_info["bit_rate"] = int(data["format"].get("bit_rate", 0))
            
            return audio_info
            
        except Exception as e:
            logging.error(f"Error getting audio info: {e}")
            return {}
    
    def normalize_audio(self, 
                       input_path: str, 
                       output_path: str,
                       target_lufs: Optional[float] = None) -> bool:
        """
        Normalize audio to target LUFS using FFmpeg.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to output audio file
            target_lufs: Target loudness in LUFS (overrides constructor setting)
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(input_path):
            logging.error(f"Input file does not exist: {input_path}")
            return False
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Use specified target LUFS or default
        lufs = target_lufs if target_lufs is not None else self.target_lufs
        
        try:
            command = [
                "ffmpeg", "-y", "-i", input_path,
                "-af", f"loudnorm=I={lufs}:TP=-1:LRA=11:print_format=summary",
                "-ar", str(self.target_sample_rate),
                output_path
            ]
            
            self._run_command(command, "Audio normalization")
            return os.path.exists(output_path)
            
        except Exception as e:
            logging.error(f"Error normalizing audio: {e}")
            return False
    
    def resample_audio(self, 
                      input_path: str, 
                      output_path: str,
                      sample_rate: Optional[int] = None) -> bool:
        """
        Resample audio to target sample rate using FFmpeg.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to output audio file
            sample_rate: Target sample rate (overrides constructor setting)
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(input_path):
            logging.error(f"Input file does not exist: {input_path}")
            return False
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Use specified sample rate or default
        target_rate = sample_rate if sample_rate is not None else self.target_sample_rate
        
        # Get current audio info
        audio_info = self.get_audio_info(input_path)
        current_rate = audio_info.get("sample_rate", 0)
        
        # Skip if already at target rate
        if current_rate == target_rate:
            logging.info(f"Audio already at target sample rate ({target_rate} Hz). Copying file.")
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            import shutil
            shutil.copy2(input_path, output_path)
            return True
        
        try:
            command = [
                "ffmpeg", "-y", "-i", input_path,
                "-ar", str(target_rate),
                output_path
            ]
            
            self._run_command(command, "Audio resampling")
            return os.path.exists(output_path)
            
        except Exception as e:
            logging.error(f"Error resampling audio: {e}")
            return False
    
    def mix_to_stereo(self, 
                     left_path: str, 
                     right_path: str, 
                     output_path: str,
                     left_pan: float = -0.2,
                     right_pan: float = 0.2,
                     instrumental_volume: float = 1.0) -> bool:
        """
        Mix left and right channels to stereo, with optional instrumental volume adjustment.
        
        Args:
            left_path: Path to left channel audio (recv_out)
            right_path: Path to right channel audio (trans_out)
            output_path: Path to output stereo audio
            left_pan: Pan value for left channel (-1.0 to 1.0)
            right_pan: Pan value for right channel (-1.0 to 1.0)
            instrumental_volume: Volume multiplier for instrumental stems (0.0 to 2.0)
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(left_path) or not os.path.exists(right_path):
            logging.error(f"Input files missing: Left: {os.path.exists(left_path)}, Right: {os.path.exists(right_path)}")
            return False
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        try:
            # Check and resample input files if needed
            left_info = self.get_audio_info(left_path)
            right_info = self.get_audio_info(right_path)
            
            # Create temporary files for resampled audio if needed
            temp_files = []
            
            # Resample left channel if needed
            left_resampled = left_path
            if left_info.get("sample_rate", 0) != self.target_sample_rate:
                left_resampled = tempfile.mktemp(suffix='.wav')
                temp_files.append(left_resampled)
                self.resample_audio(left_path, left_resampled)
            
            # Resample right channel if needed
            right_resampled = right_path
            if right_info.get("sample_rate", 0) != self.target_sample_rate:
                right_resampled = tempfile.mktemp(suffix='.wav')
                temp_files.append(right_resampled)
                self.resample_audio(right_path, right_resampled)
            
            # Build the complex filter for mixing with adjustable volumes
            filter_complex = (
                f"[0:a]pan=stereo|c0={left_pan}*c0|c1={left_pan}*c0[left];"
                f"[1:a]volume={instrumental_volume},pan=stereo|c0={right_pan}*c0|c1={right_pan}*c0[right];"
                f"[left][right]amix=inputs=2:duration=longest[out]"
            )
            
            # Run the mixing command
            command = [
                "ffmpeg", "-y", 
                "-i", left_resampled,
                "-i", right_resampled,
                "-filter_complex", filter_complex,
                "-map", "[out]",
                output_path
            ]
            
            self._run_command(command, "Stereo mixing")
            
            # Clean up temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            return os.path.exists(output_path)
            
        except Exception as e:
            logging.error(f"Error mixing to stereo: {e}")
            return False
    
    def append_tones(self, input_path: str, output_path: str, tones_path: str) -> bool:
        """
        Append tones to the end of an audio file.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to output audio file
            tones_path: Path to tones audio file
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(input_path) or not os.path.exists(tones_path):
            logging.error(f"Input files missing: Input: {os.path.exists(input_path)}, Tones: {os.path.exists(tones_path)}")
            return False
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        try:
            # Create concatenation filter
            filter_complex = "[0:a][1:a]concat=n=2:v=0:a=1[out]"
            
            # Run the concatenation command
            command = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-i", tones_path,
                "-filter_complex", filter_complex,
                "-map", "[out]",
                output_path
            ]
            
            self._run_command(command, "Appending tones")
            return os.path.exists(output_path)
            
        except Exception as e:
            logging.error(f"Error appending tones: {e}")
            return False

if __name__ == "__main__":
    # Simple CLI for testing
    import argparse
    parser = argparse.ArgumentParser(description="Process audio files")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Normalize command
    normalize_parser = subparsers.add_parser("normalize", help="Normalize audio")
    normalize_parser.add_argument("input_file", help="Input audio file")
    normalize_parser.add_argument("output_file", help="Output audio file")
    normalize_parser.add_argument("--lufs", type=float, default=-6.0, help="Target LUFS")
    
    # Resample command
    resample_parser = subparsers.add_parser("resample", help="Resample audio")
    resample_parser.add_argument("input_file", help="Input audio file")
    resample_parser.add_argument("output_file", help="Output audio file")
    resample_parser.add_argument("--rate", type=int, default=44100, help="Target sample rate")
    
    # Mix command
    mix_parser = subparsers.add_parser("mix", help="Mix to stereo")
    mix_parser.add_argument("left_file", help="Left channel audio file")
    mix_parser.add_argument("right_file", help="Right channel audio file")
    mix_parser.add_argument("output_file", help="Output audio file")
    mix_parser.add_argument("--left-pan", type=float, default=-0.2, help="Left channel pan")
    mix_parser.add_argument("--right-pan", type=float, default=0.2, help="Right channel pan")
    mix_parser.add_argument("--inst-vol", type=float, default=1.0, help="Instrumental volume")
    
    # Append tones command
    tones_parser = subparsers.add_parser("append-tones", help="Append tones")
    tones_parser.add_argument("input_file", help="Input audio file")
    tones_parser.add_argument("tones_file", help="Tones audio file")
    tones_parser.add_argument("output_file", help="Output audio file")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get audio info")
    info_parser.add_argument("input_file", help="Input audio file")
    
    args = parser.parse_args()
    
    processor = AudioProcessor()
    
    if args.command == "normalize":
        if processor.normalize_audio(args.input_file, args.output_file, args.lufs):
            print(f"Normalized audio saved to {args.output_file}")
            
    elif args.command == "resample":
        if processor.resample_audio(args.input_file, args.output_file, args.rate):
            print(f"Resampled audio saved to {args.output_file}")
            
    elif args.command == "mix":
        if processor.mix_to_stereo(args.left_file, args.right_file, args.output_file, 
                                   args.left_pan, args.right_pan, args.inst_vol):
            print(f"Mixed audio saved to {args.output_file}")
            
    elif args.command == "append-tones":
        if processor.append_tones(args.input_file, args.output_file, args.tones_file):
            print(f"Audio with appended tones saved to {args.output_file}")
            
    elif args.command == "info":
        info = processor.get_audio_info(args.input_file)
        print("\nAudio Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    else:
        parser.print_help() 