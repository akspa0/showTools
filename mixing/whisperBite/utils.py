import os
import logging
import zipfile
import subprocess
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
            # Exclude raw 'speakers' folder
            if "speakers" in root or "normalized" in root or "demucs" in root:
                continue

            # Include only files from transcriptions folders
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

def slice_on_voice_activity(input_audio, output_dir, min_silence_len=1000, silence_thresh=-40, min_duration=1000):
    """Slice audio based on voice activity detection."""
    try:
        from pydub.silence import detect_nonsilent
        logging.info(f"Starting voice activity-based slicing for {input_audio}")
        audio = AudioSegment.from_file(input_audio)
        
        # Split on silence (non-voice segments)
        chunks = detect_nonsilent(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            seek_step=10
        )
        
        segments = []
        for i, (start, end) in enumerate(chunks):
            # Only keep segments longer than min_duration
            if end - start >= min_duration:
                segment = audio[start:end]
                segment_path = os.path.join(output_dir, f"voice_segment_{i}_{start}_{end}.wav")
                segment.export(segment_path, format="wav")
                segments.append(segment_path)
                logging.info(f"Saved voice segment: {segment_path}")
        
        logging.info(f"Audio split into {len(segments)} voice segments")
        return segments
    except Exception as e:
        logging.error(f"Error during voice activity slicing: {e}")
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

def slice_by_word_timestamps(audio_file, timestamps, output_dir):
    """Slice audio file into word-level segments using Whisper timestamps."""
    try:
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
            
        logging.info(f"Starting word-level slicing for {audio_file}")
        audio = AudioSegment.from_file(audio_file)
        segments = []

        # Create a directory for word-level segments
        words_dir = os.path.join(output_dir, "words")
        os.makedirs(words_dir, exist_ok=True)

        for segment in timestamps:
            # Process each word in the segment if available
            if 'words' in segment:
                for word in segment['words']:
                    try:
                        start_ms = int(float(word['start']) * 1000)
                        end_ms = int(float(word['end']) * 1000)
                        word_text = word['text'].strip()

                        if word_text and start_ms < end_ms and end_ms <= len(audio):
                            # Extract the word segment
                            word_segment = audio[start_ms:end_ms]
                            
                            # Create filename using timestamp and word
                            safe_word = sanitize_filename(word_text)
                            segment_path = os.path.join(words_dir, 
                                f"word_{safe_word}_{start_ms}_{end_ms}.wav")
                            
                            # Export the segment
                            word_segment.export(segment_path, format="wav")
                            segments.append(segment_path)
                            logging.info(f"Saved word segment: {segment_path}")
                        else:
                            logging.warning(f"Skipping invalid word segment: {word_text} ({start_ms}ms to {end_ms}ms)")
                    except (KeyError, ValueError) as e:
                        logging.error(f"Error processing word timestamp: {e}")
                        continue

        logging.info(f"Audio split into {len(segments)} word segments")
        return segments
    except FileNotFoundError as e:
        logging.error(str(e))
        raise
    except Exception as e:
        logging.error(f"Error during word-level slicing: {e}")
        raise

def normalize_audio_ffmpeg(input_audio):
    """Normalize audio to -6 dB true peak using FFmpeg."""
    try:
        output_audio = os.path.splitext(input_audio)[0] + "_normalized.wav"
        command = [
            "ffmpeg", "-i", input_audio, "-af", "loudnorm=I=-6:TP=-1.5:LRA=11",
            "-ar", "44100", "-ac", "2", output_audio
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info(f"Normalized audio saved to {output_audio}")
        return output_audio
    except Exception as e:
        logging.error(f"Error normalizing {input_audio} with FFmpeg: {e}")
        raise
