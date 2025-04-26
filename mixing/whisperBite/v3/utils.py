import os
import logging
import zipfile
import subprocess
import shutil
from datetime import datetime
from pydub import AudioSegment
import yt_dlp
import json

def sanitize_filename(name, max_length=128):
    """Sanitize filename by removing unwanted characters and limiting length."""
    sanitized = "".join(char if char.isalnum() or char in " _-" else "_" for char in name)[:max_length]
    # Replace multiple consecutive underscores with a single one
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip()

def download_audio(url, output_dir, force_redownload=True):
    """Download audio from a URL and save it to the output directory with a unique filename."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': False,
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s')
    }

    # Add a timestamp to ensure uniqueness if forcing redownload
    if force_redownload:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ydl_opts['outtmpl'] = os.path.join(output_dir, f'%(title)s_{timestamp}.%(ext)s')

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # The file will be saved with the .wav extension due to the postprocessor
            downloaded_file = os.path.join(output_dir, f"{info['title']}.wav")
            
            if not os.path.exists(downloaded_file):
                # If the postprocessor didn't work as expected, try to find the downloaded file
                for file in os.listdir(output_dir):
                    if info['title'] in file:
                        downloaded_file = os.path.join(output_dir, file)
                        break
                
            logging.info(f"Downloaded audio saved to {downloaded_file}")
            return downloaded_file
    except Exception as e:
        logging.error(f"Error downloading audio from {url}: {e}")
        raise

def zip_results(output_dir, input_filename):
    """Zip the results directory contents into a single zip file."""
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    # Place zip file one level *above* the specific output_dir if possible, 
    # otherwise it gets included in the walk.
    parent_dir = os.path.dirname(output_dir)
    zip_filename = os.path.join(parent_dir, f"{base_name}_results_{os.path.basename(output_dir)}.zip") # Add timestamped folder name for uniqueness

    # Ensure parent directory exists (e.g., if output_dir was created at top level)
    os.makedirs(parent_dir, exist_ok=True)

    logging.info(f"Creating zip archive: {zip_filename}")

    # Folders to exclude from the zip
    excluded_folders = ["normalized", "downloads"]
    # Files to exclude (like the zip itself if it ended up in the output_dir)
    excluded_files = [os.path.basename(zip_filename)] 

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            # Modify dirs in place to prevent walking into excluded folders
            dirs[:] = [d for d in dirs if d not in excluded_folders]

            for file in files:
                if file in excluded_files:
                    continue
                
                full_path = os.path.join(root, file)
                # Archive name is path relative to output_dir
                archive_name = os.path.relpath(full_path, output_dir)
                
                logging.debug(f"Adding to zip: {full_path} as {archive_name}")
                zipf.write(full_path, archive_name)

        # --- Optionally, add metadata --- 
        # (Can be refined later based on what's useful)
        try:
            metadata = {
                "source_file": os.path.basename(input_filename),
                "processing_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                # Add more metadata if needed, e.g., parameters used
            }
            from io import BytesIO
            metadata_bytes = BytesIO(json.dumps(metadata, indent=2).encode('utf-8'))
            zipf.writestr("processing_metadata.json", metadata_bytes.getvalue())
        except Exception as meta_err:
             logging.warning(f"Could not generate or add metadata.json: {meta_err}")
        # --- End metadata --- 

    logging.info(f"Successfully created zip file: {zip_filename}")
    return zip_filename

def create_thumbnail_waveform(audio_file, output_path):
    """Create a waveform thumbnail image for the audio file."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from pydub import AudioSegment
        
        # Load audio
        audio = AudioSegment.from_file(audio_file)
        samples = np.array(audio.get_array_of_samples())
        
        # Normalize
        samples = samples / np.max(np.abs(samples))
        
        # Create figure with transparent background
        fig, ax = plt.subplots(figsize=(5, 1))
        fig.patch.set_alpha(0)
        ax.set_alpha(0)
        
        # Plot waveform
        ax.plot(samples, color='blue', linewidth=0.5)
        ax.axis('off')
        
        # Save
        plt.tight_layout()
        plt.savefig(output_path, transparent=True, dpi=100)
        plt.close()
        
        return output_path
    except Exception as e:
        logging.warning(f"Error creating waveform thumbnail: {e}")
        return None
