import os
import re

def search_files(directory):
    cue_files = []
    audio_files = []
    artwork_files = []
    log_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            lower_file = file.lower()
            if lower_file.endswith('.cue'):
                cue_files.append(os.path.join(root, file))
            elif lower_file.endswith(('.flac', '.wav')):
                audio_files.append(os.path.join(root, file))
            elif lower_file.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                artwork_files.append(os.path.join(root, file))
            elif lower_file.endswith('.log'):
                log_files.append(os.path.join(root, file))

    return cue_files, audio_files, artwork_files, log_files

def get_relative_path(path, base_path):
    """Get the relative path from base_path to path"""
    return os.path.relpath(path, base_path)

def create_output_structure(input_path, output_base):
    """Create the output directory structure mirroring the input path"""
    try:
        # Try to get relative path, but handle different drives
        rel_path = os.path.relpath(input_path, output_base)
    except ValueError:  # Different drives
        # Preserve the full path structure without drive letter
        drive, path = os.path.splitdrive(input_path)
        rel_path = path.lstrip(os.sep)
    
    # Create main output directory
    output_path = os.path.join(output_base, rel_path)
    os.makedirs(output_path, exist_ok=True)
    
    # Create artwork subdirectory
    artwork_path = os.path.join(output_path, 'artwork')
    os.makedirs(artwork_path, exist_ok=True)
    
    return output_path, artwork_path

def copy_support_files(files, input_base, output_base):
    """Copy artwork and log files while maintaining directory structure"""
    import shutil
    
    for file_path in files:
        # Get the relative path from input base
        rel_path = os.path.relpath(file_path, input_base)
        
        # Determine if this is artwork
        is_artwork = any(file_path.lower().endswith(ext)
                        for ext in ('.jpg', '.jpeg', '.png', '.gif'))
        
        # For artwork, place in artwork subdirectory
        if is_artwork:
            rel_dir = os.path.dirname(rel_path)
            output_dir = os.path.join(output_base, rel_dir, 'artwork')
        else:
            # For other files (like logs), maintain exact structure
            output_dir = os.path.join(output_base, os.path.dirname(rel_path))
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy the file
        output_file = os.path.join(output_dir, os.path.basename(file_path))
        shutil.copy2(file_path, output_file)
        print(f"Copied {'artwork' if is_artwork else 'support file'}: {output_file}")

def find_matching_audio(cue_file, audio_files):
    # First try to find the audio file specified in the CUE file
    with open(cue_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip().startswith('FILE '):
                # Extract filename from the FILE directive
                match = re.search(r'FILE\s+"([^"]+)"', line)
                if match:
                    specified_file = match.group(1)
                    # Look for this exact file in audio_files
                    for audio_file in audio_files:
                        if os.path.basename(audio_file) == specified_file:
                            return audio_file

    # Fall back to base name comparison if exact match not found
    cue_base_name = os.path.splitext(os.path.basename(cue_file))[0].strip()
    cue_base_name_regex = re.compile(re.escape(cue_base_name), re.IGNORECASE)

    for audio_file in audio_files:
        audio_base_name = os.path.splitext(os.path.basename(audio_file))[0].strip()
        print(f'Comparing {cue_base_name} with {audio_base_name}')
        if cue_base_name_regex.search(audio_base_name):
            return audio_file

    return None