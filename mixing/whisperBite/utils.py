import os
import logging
import zipfile
from pydub import AudioSegment
import yt_dlp

def sanitize_filename(name, max_length=128):
    """Sanitize filename by removing unwanted characters and limiting length."""
    sanitized = "".join(char if char.isalnum() or char == "_" else "_" for char in name)[:max_length]
    return sanitized

def download_audio(url, output_dir):
    """Download audio from a URL and save it to the output directory with a unique filename."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': False,
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s')  # Use the video title as the filename
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_file = os.path.join(output_dir, f"{info['title']}.{info['ext']}")
            logging.info(f"Downloaded audio saved to {downloaded_file}")
            return downloaded_file
    except Exception as e:
        logging.error(f"Error downloading audio from {url}: {e}")
        raise

def zip_results(output_dir, input_filename):
    """Zip the results into a single zip file containing only speaker transcriptions and their audio files."""
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    zip_filename = os.path.join(output_dir, f"{base_name}_results.zip")

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            if "normalized" in root or "demucs" in root:
                continue  # Skip normalized and demucs directories

            for file in files:
                if file.endswith(".txt") or file.endswith(".wav"):
                    full_path = os.path.join(root, file)

                    # Convert WAV to MP3 before adding to the zip
                    if file.endswith(".wav"):
                        mp3_file = os.path.splitext(full_path)[0] + ".mp3"
                        audio = AudioSegment.from_file(full_path)
                        audio.export(mp3_file, format="mp3")

                        # Add the MP3 to the zip
                        relative_path = os.path.relpath(mp3_file, output_dir)
                        zipf.write(mp3_file, relative_path)

                        # Clean up the MP3 file after adding it to the zip
                        os.remove(mp3_file)
                    else:
                        # Add other files (e.g., .txt) directly
                        relative_path = os.path.relpath(full_path, output_dir)
                        zipf.write(full_path, relative_path)

    logging.info(f"Results zipped into: {zip_filename}")
    return zip_filename

def convert_and_normalize_audio(input_audio, output_dir):
    """Convert audio to WAV with 44.1kHz, 16-bit and normalize loudness to -14 LUFS."""
    try:
        logging.info(f"Processing audio file: {input_audio}")
        normalized_dir = os.path.join(output_dir, "normalized")
        os.makedirs(normalized_dir, exist_ok=True)

        audio = AudioSegment.from_file(input_audio)

        # Set sample rate and bit depth (16-bit PCM)
        audio = audio.set_frame_rate(44100).set_sample_width(2)

        # Normalize loudness to -14 LUFS
        change_in_dBFS = -14 - audio.dBFS
        normalized_audio = audio.apply_gain(change_in_dBFS)

        # Export normalized audio
        base_name = os.path.splitext(os.path.basename(input_audio))[0]
        output_file = os.path.join(normalized_dir, f"{base_name}_normalized.wav")
        normalized_audio.export(output_file, format="wav", parameters=["-acodec", "pcm_s16le"])
        logging.info(f"Normalized audio saved to {output_file}")
        return output_file
    except Exception as e:
        logging.error(f"Error normalizing {input_audio}: {e}")
        raise

def default_slicing(input_audio, output_dir, min_duration=5000, silence_thresh=-40):
    """Slice audio into smaller chunks based on silence or duration."""
    try:
        logging.info(f"Starting default slicing for {input_audio}")
        audio = AudioSegment.from_file(input_audio)
        segments = []
        start = 0

        while start < len(audio):
            end = start + min_duration
            segment = audio[start:end]

            # Export the segment
            segment_path = os.path.join(output_dir, f"soundbite_{start // 1000}_{end // 1000}.wav")
            segment.export(segment_path, format="wav")
            segments.append(segment_path)

            start = end

        logging.info(f"Audio split into {len(segments)} soundbites.")
        return segments
    except Exception as e:
        logging.error(f"Error during default slicing: {e}")
        raise
