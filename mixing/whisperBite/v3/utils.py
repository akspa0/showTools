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
    """Zip the results into a single zip file containing transcriptions and their audio files."""
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    zip_filename = os.path.join(output_dir, f"{base_name}_results.zip")

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add the master transcript
        master_transcript = os.path.join(output_dir, "master_transcript.txt")
        if os.path.exists(master_transcript):
            zipf.write(master_transcript, os.path.basename(master_transcript))
        
        # Add speaker transcripts
        for item in os.listdir(output_dir):
            if item.startswith("Speaker_") and item.endswith("_full_transcript.txt"):
                full_path = os.path.join(output_dir, item)
                zipf.write(full_path, item)
        
        # Add individual transcriptions and their audio files
        for root, dirs, files in os.walk(output_dir):
            # Skip processing directories
            if any(x in root for x in ["normalized", "demucs", "speakers"]):
                continue
                
            # Include transcriptions directories
            if "_transcriptions" in root:
                for file in files:
                    full_path = os.path.join(root, file)
                    
                    # Store folder structure relative to output_dir
                    relative_path = os.path.relpath(full_path, output_dir)
                    
                    # Add .txt files directly
                    if file.endswith(".txt"):
                        zipf.write(full_path, relative_path)
                    
                    # Convert WAV to MP3 for audio files to save space
                    elif file.endswith(".wav"):
                        try:
                            mp3_file = os.path.splitext(full_path)[0] + ".mp3"
                            audio = AudioSegment.from_file(full_path)
                            audio.export(mp3_file, format="mp3", bitrate="128k")
                            
                            # Add MP3 to the zip with the same relative path but different extension
                            mp3_relative_path = os.path.splitext(relative_path)[0] + ".mp3"
                            zipf.write(mp3_file, mp3_relative_path)
                            
                            # Clean up the MP3 file after adding to zip
                            os.remove(mp3_file)
                        except Exception as e:
                            logging.warning(f"Error converting {file} to MP3: {e}. Adding WAV instead.")
                            zipf.write(full_path, relative_path)
        
        # Add segment information JSON
        segments_json = os.path.join(output_dir, "speakers", "segments.json")
        if os.path.exists(segments_json):
            zipf.write(segments_json, "segments.json")
            
        # Create a metadata.json file with processing information
        metadata = {
            "source_file": os.path.basename(input_filename),
            "processing_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "num_speakers": len([d for d in os.listdir(output_dir) if d.startswith("Speaker_") and d.endswith("_full_transcript.txt")]),
        }
        
        # Add metadata to zip
        from io import BytesIO
        metadata_bytes = BytesIO(json.dumps(metadata, indent=2).encode('utf-8'))
        zipf.writestr("metadata.json", metadata_bytes.getvalue())

    logging.info(f"Results zipped into: {zip_filename}")
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
