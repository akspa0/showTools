import logging
import shutil
import re
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)

def ensure_dir(dir_path: Path):
    """Creates a directory if it doesn't exist, including parent directories."""
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        log.debug(f"Ensured directory exists: {dir_path}")
    except OSError as e:
        log.error(f"Failed to create directory {dir_path}: {e}")
        raise # Re-raise the exception after logging

def sanitize_filename(filename: str) -> str:
    """Removes or replaces characters invalid for directory/file names."""
    # Remove leading/trailing whitespace
    sanitized = filename.strip()
    # Remove control characters
    sanitized = "".join(c for c in sanitized if ord(c) >= 32)
    # Replace potentially problematic characters with underscore
    sanitized = re.sub(r'[<>:"/\\|?*\.]+', '_', sanitized) # Added dot to avoid issues with hidden files or extension removal
    # Replace multiple consecutive underscores with a single one
    sanitized = re.sub(r'_+', '_', sanitized)
    # Limit length if necessary (optional)
    # max_len = 100
    # sanitized = sanitized[:max_len]
    if not sanitized:
        sanitized = "invalid_filename"
    log.debug(f'Sanitized "{filename}" to "{sanitized}"')
    return sanitized

def generate_output_path(base_dir: Path, input_filename: str) -> Path:
    """Creates a unique, timestamped output directory based on the input filename."""
    ensure_dir(base_dir) # Ensure base output directory exists

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get filename stem and sanitize it
    input_path = Path(input_filename)
    sanitized_stem = sanitize_filename(input_path.stem) # Sanitize only the stem
    
    # Combine to create unique directory name
    output_dir_name = f"{sanitized_stem}_{timestamp}"
    output_path = base_dir / output_dir_name
    
    ensure_dir(output_path) # Create the specific output directory
    log.info(f"Generated output path: {output_path}")
    return output_path

def cleanup_directory(dir_path: Path):
    """Recursively removes a directory and its contents."""
    if not dir_path.exists() or not dir_path.is_dir():
        log.warning(f"Attempted to clean up non-existent or non-directory path: {dir_path}")
        return
    try:
        shutil.rmtree(dir_path)
        log.info(f"Successfully cleaned up directory: {dir_path}")
    except OSError as e:
        log.error(f"Failed to clean up directory {dir_path}: {e}")
        # Decide if we should raise here or just log
        # raise e 