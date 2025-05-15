import logging
from pathlib import Path
from typing import List, Dict, Optional

# Attempt to import the Separator class
try:
    from audio_separator.separator import Separator
except ImportError:
    Separator = None # Set to None if library is not installed
    logging.error("Failed to import Separator from audio_separator. Please ensure the library is installed.")

from config import settings
from utils.file_utils import ensure_dir

log = logging.getLogger(__name__)

class AudioSeparatorWrapper:
    """A wrapper class for the python-audio-separator library."""

    def __init__(self,
                 model_name: str,
                 output_dir: Path = settings.TEMP_OUTPUT_DIR,
                 model_file_dir: Path = settings.AUDIO_SEPARATOR_MODEL_DIR,
                 log_level: str = settings.LOG_LEVEL,
                 **separator_options):
        """Initializes the audio separator wrapper.

        Args:
            model_name: The name of the separation model to use (must be valid for audio-separator).
            output_dir: Directory to save intermediate separated files.
            model_file_dir: Directory to cache downloaded models.
            log_level: Logging level for the separator.
            **separator_options: Additional keyword arguments passed directly to the Separator constructor
                                 (e.g., mdx_params, vr_params, demucs_params, use_soundfile).
        """
        if Separator is None:
            raise RuntimeError("audio_separator library not found or failed to import.")
        
        self.model_name = model_name
        # Remove .ckpt extension from model name for output filename matching
        self.model_name_base = model_name.replace('.ckpt', '')
        self.output_dir = output_dir
        self.model_file_dir = model_file_dir
        self.separator_options = separator_options
        
        ensure_dir(self.output_dir)
        ensure_dir(self.model_file_dir)

        # Default options recommended or useful
        self.separator_options.setdefault('use_soundfile', True)
        
        # Remove any parameters that are no longer supported by the Separator constructor
        for param in ['use_cuda']:
            if param in self.separator_options:
                log.warning(f"Removing unsupported parameter '{param}' from separator options")
                self.separator_options.pop(param)

        log.info(f"Initializing Separator for model: {self.model_name}")
        log.info(f"Output directory: {self.output_dir}")
        log.debug(f"Separator options: {self.separator_options}")

        try:
            # Initialize the separator with basic parameters
            self.separator = Separator(
                output_dir=str(self.output_dir),
                model_file_dir=str(self.model_file_dir)
            )
            log.info("Separator initialized successfully.")
            
            # Load the specified model
            log.info(f"Loading model: {self.model_name}")
            self.separator.load_model(self.model_name)
            log.info(f"Model {self.model_name} loaded successfully")
                
        except Exception as e:
            log.exception(f"Failed to initialize Separator: {e}")
            raise RuntimeError(f"Failed to initialize Separator: {e}") from e

    def separate(self, input_audio_path: Path) -> Dict[str, Path]:
        """Separates the input audio into 'Vocals' and 'Instrumental' stems.

        Args:
            input_audio_path: Path to the input audio file.

        Returns:
            A dictionary mapping stem names ('Vocals', 'Instrumental') to their output file paths.
            Returns an empty dictionary if separation fails or doesn't produce expected stems.

        Raises:
            FileNotFoundError: If the input audio file does not exist.
            RuntimeError: If the separation process encounters an error.
        """
        if not input_audio_path.is_file():
            log.error(f"Input audio file not found: {input_audio_path}")
            raise FileNotFoundError(f"Input audio file not found: {input_audio_path}")

        log.info(f"Starting separation for: {input_audio_path} using model {self.model_name}")
        log.info(f"Output will be saved to: {self.output_dir}")

        # Define expected output stem names for later mapping
        # This helps identify the output files after separation
        # Use model_name_base (without .ckpt) for output filename matching
        expected_stems = {
            "Vocals": f"{input_audio_path.stem}_(Vocals)_{self.model_name_base}",
            "Instrumental": f"{input_audio_path.stem}_(Instrumental)_{self.model_name_base}"
            # Add other stems here if needed, e.g., "Drums", "Bass"
        }

        try:
            # Separate the audio file - removed output_names parameter
            output_files = self.separator.separate(
                str(input_audio_path)
            )
                
            log.info(f"Separation completed. Output files: {output_files}")

            # Verify and map the expected output files
            result_paths: Dict[str, Path] = {}
            
            # Check if output_files contains full paths or just filenames
            if output_files and isinstance(output_files[0], str) and '/' not in output_files[0] and '\\' not in output_files[0]:
                # If just filenames, prepend the output directory
                output_files_full = [self.output_dir / Path(f).name for f in output_files]
                log.info(f"Converted to full paths: {output_files_full}")
                output_files_set = {Path(f).name for f in output_files}
            else:
                # If full paths, use as is
                output_files_full = [Path(f) for f in output_files]
                output_files_set = {Path(f).name for f in output_files}

            # Map output files to expected stems
            for stem_name, expected_filename_base in expected_stems.items():
                # Construct potential filename (separator might add extension)
                # Check common extensions
                found = False
                for ext in [".wav", ".mp3", ".flac", ".m4a"]: # Add more if needed
                    expected_filename = f"{expected_filename_base}{ext}"
                    if expected_filename in output_files_set:
                        # Find the corresponding full path
                        for full_path in output_files_full:
                            if full_path.name == expected_filename:
                                result_paths[stem_name] = full_path
                                log.debug(f"Found expected stem '{stem_name}' at: {result_paths[stem_name]}")
                                found = True
                                break
                        if found:
                            break
                if not found:
                     log.warning(f"Could not find expected output file for stem: '{stem_name}' (expected base: {expected_filename_base})")
            
            if not result_paths:
                log.error("Separation finished but no expected output stems (Vocals/Instrumental) were found.")
                # Consider raising an error or returning empty dict based on desired strictness

            return result_paths

        except Exception as e:
            log.exception(f"Audio separation failed for {input_audio_path}: {e}")
            raise RuntimeError(f"Audio separation failed: {e}") from e 