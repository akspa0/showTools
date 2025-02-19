"""
Demucs processing module for vocal separation.
Handles the core Demucs vocal separation.
"""

import os
import shutil
import logging
import subprocess
from typing import Optional
from whisperBite.config import AudioProcessingError

logger = logging.getLogger(__name__)

class DemucsProcessor:
    """Handles Demucs vocal separation processing."""
    
    def __init__(self):
        self.model = "htdemucs"
        self.shifts = 2
        self.overlap = 0.25
    
    def process(self, input_path: str, output_path: str) -> str:
        """Run Demucs vocal separation."""
        try:
            # Run Demucs
            self._run_demucs(input_path)
            
            # Find and move output
            temp_output = self._find_output(input_path)
            if temp_output and os.path.exists(temp_output):
                # Use shutil.copy2 instead of os.rename for cross-device moves
                shutil.copy2(temp_output, output_path)
                # Clean up the temporary file
                try:
                    os.remove(temp_output)
                    # Also try to clean up the parent directories if empty
                    os.removedirs(os.path.dirname(temp_output))
                except OSError:
                    # Ignore errors during cleanup
                    pass
                return output_path
            else:
                raise AudioProcessingError("Could not find Demucs output")
                
        except Exception as e:
            logger.error(f"Demucs processing failed: {str(e)}")
            raise AudioProcessingError(f"Demucs processing failed: {str(e)}")
    
    def _run_demucs(self, input_path: str):
        """Execute Demucs command."""
        try:
            command = [
                "demucs",
                "--two-stems", "vocals",
                "-n", self.model,
                "--shifts", str(self.shifts),
                "--overlap", str(self.overlap),
                input_path
            ]
            
            # Log the exact command being run
            logger.info(f"Running Demucs command: {' '.join(command)}")
            
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True
            )
            
            # Log any output
            if result.stdout:
                logger.info(f"Demucs stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"Demucs stderr: {result.stderr}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Demucs command failed: {' '.join(command)}")
            logger.error(f"Demucs error output: {e.stderr}")
            raise AudioProcessingError(f"Demucs execution failed: {e.stderr}")
    
    def _find_output(self, input_path: str) -> Optional[str]:
        """Locate Demucs output file."""
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        potential_paths = [
            os.path.join("separated", "htdemucs", base_name, "vocals.wav"),
            os.path.join("separated", "mdx_extra", base_name, "vocals.wav")
        ]
        
        logger.debug(f"Looking for Demucs output in potential paths: {potential_paths}")
        
        for path in potential_paths:
            if os.path.exists(path):
                logger.info(f"Found Demucs output at: {path}")
                return path
        
        logger.warning(f"No Demucs output found in any of the potential paths")
        return None
