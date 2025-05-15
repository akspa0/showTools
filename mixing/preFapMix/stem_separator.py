import os
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StemSeparator:
    """
    Interface to python-audio-separator for separating vocals from instrumentals.
    
    This class provides methods to use python-audio-separator to split audio into
    vocal and instrumental stems, with error handling and logging.
    """
    
    def __init__(self, 
                 model_name: str = "UVR-MDX-NET-Voc_FT",
                 output_dir: Optional[str] = None,
                 model_file_dir: Optional[str] = None):
        """
        Initialize the stem separator.
        
        Args:
            model_name: Name of the separation model to use 
                (default: "UVR-MDX-NET-Voc_FT")
            output_dir: Directory to save separated stems
                (default: creates a temporary directory)
            model_file_dir: Directory to cache model files
                (default: uses audio-separator's default cache)
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.model_file_dir = model_file_dir
        
        # Check if audio-separator is installed
        try:
            self._run_command(["audio-separator", "--version"], "Checking audio-separator")
        except Exception as e:
            logging.error(f"audio-separator not found. Please install it with 'pip install audio-separator'")
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
        
    def separate(self, 
                 input_path: str, 
                 output_dir: Optional[str] = None,
                 custom_output_names: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Separate vocals from instrumentals.
        
        Args:
            input_path: Path to input audio file
            output_dir: Directory to save output (overrides constructor setting)
            custom_output_names: Optional dictionary to customize output filenames
                (e.g., {"Vocals": "custom_vocals", "Instrumental": "custom_instrumental"})
                
        Returns:
            Dictionary mapping stem types to output file paths
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
        command = ["audio-separator", input_path, "-o", target_output_dir, "-m", self.model_name]
        
        # Add model file directory if specified
        if self.model_file_dir:
            command.extend(["--model_file_dir", self.model_file_dir])
        
        # Add custom output names if provided
        if custom_output_names:
            for stem_type, custom_name in custom_output_names.items():
                command.extend(["--output_format", f"{stem_type}={custom_name}.wav"])
        
        # Run the separation command
        try:
            self._run_command(command, "Audio separation")
            
            # Determine the output file paths
            base_name = input_file.stem
            result = {}
            
            # Expected output pattern based on audio-separator behavior
            if custom_output_names and "Vocals" in custom_output_names:
                vocals_name = f"{custom_output_names['Vocals']}.wav"
                vocals_path = os.path.join(target_output_dir, vocals_name)
                if os.path.exists(vocals_path):
                    result["Vocals"] = vocals_path
            else:
                # Default naming pattern
                vocals_path = os.path.join(target_output_dir, f"{base_name}_(Vocals)_{self.model_name}.wav")
                if os.path.exists(vocals_path):
                    result["Vocals"] = vocals_path
            
            if custom_output_names and "Instrumental" in custom_output_names:
                inst_name = f"{custom_output_names['Instrumental']}.wav"
                inst_path = os.path.join(target_output_dir, inst_name)
                if os.path.exists(inst_path):
                    result["Instrumental"] = inst_path
            else:
                # Default naming pattern
                inst_path = os.path.join(target_output_dir, f"{base_name}_(Instrumental)_{self.model_name}.wav")
                if os.path.exists(inst_path):
                    result["Instrumental"] = inst_path
            
            # Check if we found expected outputs
            if not result:
                logging.warning(f"No output files found in {target_output_dir}")
                # Try to find any possible output files
                for file in os.listdir(target_output_dir):
                    if file.endswith(".wav"):
                        if "vocals" in file.lower():
                            result["Vocals"] = os.path.join(target_output_dir, file)
                        elif "instrumental" in file.lower():
                            result["Instrumental"] = os.path.join(target_output_dir, file)
            
            logging.info(f"Separation completed. Found stems: {list(result.keys())}")
            return result
            
        except Exception as e:
            logging.error(f"Error during separation: {e}")
            return {}

def list_available_models() -> List[str]:
    """List available separation models in audio-separator."""
    try:
        result = subprocess.run(
            ["audio-separator", "--list-models"],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Parse model names from output
        models = []
        for line in result.stdout.splitlines():
            if line.strip() and not line.startswith("Available models:"):
                models.append(line.strip())
        
        return models
    except Exception as e:
        logging.error(f"Error listing models: {e}")
        return []

if __name__ == "__main__":
    # Simple CLI for testing
    import argparse
    parser = argparse.ArgumentParser(description="Separate vocals from instrumentals")
    parser.add_argument("input_file", help="Input audio file")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--model", default="UVR-MDX-NET-Voc_FT", help="Separation model")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    args = parser.parse_args()
    
    if args.list_models:
        models = list_available_models()
        print("\nAvailable separation models:")
        for model in models:
            print(f"  - {model}")
        exit(0)
    
    separator = StemSeparator(model_name=args.model, output_dir=args.output_dir)
    output_files = separator.separate(args.input_file)
    
    print("\nSeparation completed:")
    for stem_type, path in output_files.items():
        print(f"  - {stem_type}: {path}") 