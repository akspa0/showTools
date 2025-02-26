import argparse
import os
import whisper
import librosa
import numpy as np
import json
import datetime
import soundfile as sf
import re
import subprocess
import sqlite3

def sanitize_filename(filename):
    """Sanitizes a filename by removing invalid characters."""
    return re.sub(r'[^\w\-_\.]', '', filename)

def generate_text_based_filename(text, counter, base_dir, extension=".wav"):
    """Generates a filename from text, counter, and base directory."""
    sanitized_text = sanitize_filename(text.strip().replace(" ", "_"))
    max_len = 40 - len(extension) - len(str(counter)) - 1  # Account for counter, extension, and separator
    truncated_text = sanitized_text[:max_len]
    filename = f"{counter:04d}_{truncated_text}{extension}"
    return os.path.join(base_dir, filename)

def detect_dtmf(audio_file):
    """Detects DTMF tones in an audio file."""
    y, sr = librosa.load(audio_file)
    hop_length = int(sr * 0.01)  # 10ms hop
    frame_length = int(sr * 0.02) # 20ms frame

    dtmf_tones = []
    onsets = []

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    for onset_time in onset_times:
        start_frame = int(librosa.time_to_frames(onset_time, sr=sr, hop_length=hop_length, n_fft=frame_length))
        end_frame = start_frame + int(0.1 * sr / hop_length)

        if end_frame * hop_length >= len(y):
          end_frame = int(len(y)/hop_length)-1

        y_segment = y[start_frame * hop_length : end_frame * hop_length]
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
        magnitudes = np.abs(np.fft.fft(y_segment, n=frame_length))[:len(frequencies)]
        magnitudes_db = librosa.amplitude_to_db(magnitudes, ref=np.max)

        dtmf_freqs = [697, 770, 852, 941, 1209, 1336, 1477, 1633]
        tolerance = 5

        dtmf_detected = False
        for freq in dtmf_freqs:
            lower_bound = freq - tolerance
            upper_bound = freq + tolerance
            peak_indices = np.where((frequencies >= lower_bound) & (frequencies <= upper_bound))[0]
            if any(magnitudes_db[peak_indices] > -30):
                dtmf_detected = True
                break

        if dtmf_detected:
            onsets.append(onset_time)

    if onsets:
      merged_onsets = [onsets[0]]
      for onset in onsets[1:]:
          if onset - merged_onsets[-1] < 0.3:
              merged_onsets[-1] = onset
          else:
              dtmf_tones.append((merged_onsets[-1]-0.1, merged_onsets[-1]))
              merged_onsets = [onset]
      dtmf_tones.append((merged_onsets[-1]-0.1, merged_onsets[-1]))

    return dtmf_tones

def calculate_audio_quality(audio_segment, sr):
    """Calculates a simple audio quality score."""
    flatness = librosa.feature.spectral_flatness(y=audio_segment)
    return np.mean(flatness)

def query_local_llm(prompt):
    """Placeholder for querying a local LLM (Ollama or llama.cpp)."""
    # This function would need to:
    # 1. Check if a local LLM server is running (Ollama or llama.cpp).
    # 2. Construct the appropriate API request (likely a POST request).
    # 3. Send the request and receive the response.
    # 4. Parse the response to extract the relevant information (paragraph breaks).
    print("Warning: Local LLM integration is not yet implemented.")
    return []  # Return an empty list as a placeholder.

def infer_paragraphs(segments, use_llm=False):
    """Infers paragraph boundaries from sentence segments."""
    if use_llm:
        # Construct a prompt for the LLM.
        prompt = "Identify the paragraph breaks in the following text:\n\n"
        for i, segment in enumerate(segments):
            prompt += f"{i+1}. {segment['text']}\n"
        prompt += "\nParagraph breaks (sentence numbers):"
        paragraph_indices = query_local_llm(prompt)
        if paragraph_indices:
            return paragraph_indices
        # else, fall through to the heuristic

    # Simple heuristic: group sentences with short pauses between them.
    paragraphs = []
    current_paragraph = []
    if segments:
      current_paragraph.append(0)
      paragraphs.append(current_paragraph)

    for i in range(len(segments) - 1):
        current_end = segments[i]["end"]
        next_start = segments[i+1]["start"]
        if next_start - current_end < 1.0:
            current_paragraph.append(i + 1)
        else:
            current_paragraph = [i + 1]
            paragraphs.append(current_paragraph)
    return paragraphs

def create_database(db_path):
    """Creates the SQLite database and tables."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                file_id INTEGER PRIMARY KEY,
                filename TEXT,
                original_filepath TEXT,
                processed_date TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paragraphs (
                paragraph_id INTEGER PRIMARY KEY,
                file_id INTEGER,
                paragraph_filename TEXT,
                paragraph_text TEXT,
                FOREIGN KEY (file_id) REFERENCES files(file_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentences (
                sentence_id INTEGER PRIMARY KEY,
                paragraph_id INTEGER,
                sentence_filename TEXT,
                sentence_text TEXT,
                start_time REAL,
                end_time REAL,
                FOREIGN KEY (paragraph_id) REFERENCES paragraphs(paragraph_id)
            )
        """)
        #  speakers and speaker_segments tables are created in diarize_audio.py

        conn.commit()
        conn.close()
        print(f"Database created/connected successfully at: {db_path}")
    except sqlite3.Error as e:
        print(f"Error creating database: {e}")
        exit(1) # Exit if database creation fails


def insert_file_data(db_path, filename, original_filepath, processed_date):
    """Inserts file data into the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO files (filename, original_filepath, processed_date) VALUES (?, ?, ?)",
                   (filename, original_filepath, processed_date))
    file_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return file_id

def insert_paragraph_data(db_path, file_id, paragraph_filename, paragraph_text):
    """Inserts paragraph data into the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO paragraphs (file_id, paragraph_filename, paragraph_text) VALUES (?, ?, ?)",
                   (file_id, paragraph_filename, paragraph_text))
    paragraph_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return paragraph_id

def insert_sentence_data(db_path, paragraph_id, sentence_filename, sentence_text, start_time, end_time):
    """Inserts sentence data into the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO sentences (paragraph_id, sentence_filename, sentence_text, start_time, end_time) VALUES (?, ?, ?, ?, ?)",
                   (paragraph_id, sentence_filename, sentence_text, start_time, end_time))
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Audio File Processor")
    parser.add_argument("folder_path", help="Path to the folder containing audio files")
    parser.add_argument("--buffer", type=float, default=0.25, help="Buffer duration in seconds (default: 0.25)")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM for paragraph inference (placeholder)")
    parser.add_argument("--split", action="store_true", help="Split vocals from audio using Demucs")
    parser.add_argument("--normalize", action="store_true", help="Normalize audio after splitting")
    parser.add_argument("--db_path", default="transcriptions.db", help="Path to the SQLite database file")

    args = parser.parse_args()
    folder_path = args.folder_path
    buffer_duration = args.buffer
    use_llm = args.use_llm
    split_vocals = args.split
    normalize_audio = args.normalize
    db_path = args.db_path
    model_name = "turbo"

    if not os.path.isdir(folder_path):
        print(f"Error: Folder path '{folder_path}' is not a valid directory.")
        return

    create_database(db_path) # Create database and tables

    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)

    print(f"Processing audio files in folder: {folder_path}")
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav") or filename.endswith(".mp3") or filename.endswith(".m4a"):
            audio_file_path = os.path.join(folder_path, filename)
            print(f"Transcribing: {audio_file_path}")

            base_filename = os.path.splitext(filename)[0]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir_name = f"{'_'.join(base_filename.split('_')[:3])}_{timestamp}"
            output_dir_path = os.path.join(folder_path, output_dir_name)
            sentences_dir_path = os.path.join(output_dir_path, "sentences")
            paragraphs_dir_path = os.path.join(output_dir_path, "paragraphs")

            os.makedirs(sentences_dir_path, exist_ok=True)
            os.makedirs(paragraphs_dir_path, exist_ok=True)

            # --- Demucs Vocal Separation (if requested) ---
            if split_vocals:
                print(f"Splitting vocals for: {filename}")
                # Create a 'separated' subdirectory within the output directory
                separated_dir = os.path.join(output_dir_path, "separated")
                os.makedirs(separated_dir, exist_ok=True)  # Ensure 'separated' dir exists
                try:
                    # Run demucs, specifying the output directory
                    subprocess.run([
                        "demucs",
                        "--two-stems", "vocals",
                        "-n", "htdemucs",
                        "--int24",
                        "-o", separated_dir,  # Output to the 'separated' dir
                        audio_file_path
                    ], check=True)

                    # Demucs output structure: {output_dir}/separated/htdemucs/{filename}/{stem}.wav
                    vocals_file_path = os.path.join(separated_dir, "htdemucs", base_filename, "vocals.wav")

                    if not os.path.exists(vocals_file_path):
                        print(f"Error: Vocal separation failed for {filename}.  Falling back to original audio.")
                        vocals_file_path = audio_file_path  # Use original if separation fails
                except FileNotFoundError:
                    print("Error: demucs not found. Please ensure it's installed and in your PATH.")
                    return
                except subprocess.CalledProcessError as e:
                    print(f"Error running demucs: {e}")
                    return
            else:
                vocals_file_path = audio_file_path

            # --- Load Audio (either original or separated vocals) ---
            y, sr = librosa.load(vocals_file_path)

            # --- Normalization (if requested) ---
            if normalize_audio:
                print(f"Normalizing audio for: {filename}")
                y = librosa.util.normalize(y)

            dtmf_times = detect_dtmf(vocals_file_path) # Use vocals file for DTMF
            result = model.transcribe(vocals_file_path, verbose=False, word_timestamps=True) # and Whisper



            # --- Insert File Data ---
            processed_date = datetime.datetime.now().isoformat()
            file_id = insert_file_data(db_path, filename, audio_file_path, processed_date)


            # --- Sentence-level processing ---
            sentence_counter = 1
            for segment in result["segments"]:
                start_time = segment["start"]
                end_time = segment["end"]

                # NO buffer added here

                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                sentence_audio = y[start_sample:end_sample]

                sentence_text = segment["text"]
                sentence_filename = generate_text_based_filename(sentence_text, sentence_counter, sentences_dir_path)
                sf.write(os.path.join(sentences_dir_path, sentence_filename), sentence_audio, sr)
                sentence_counter += 1




            # --- Paragraph-level processing ---
            paragraphs = infer_paragraphs(result["segments"], use_llm=use_llm)

            paragraph_counter = 1
            for paragraph_indices in paragraphs:
                # Combine audio segments for the paragraph
                paragraph_audio = np.array([], dtype=np.float32)
                paragraph_text = "" # Accumulate text for filename
                for i in paragraph_indices:
                    segment = result["segments"][i]
                    start_time = segment["start"]
                    end_time = segment["end"]
                    # NO buffer added here
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    paragraph_audio = np.concatenate((paragraph_audio, y[start_sample:end_sample]))
                    paragraph_text += segment["text"] + " " # Accumulate text

                # Add buffer AFTER combining
                end_time = min(librosa.get_duration(y=y, sr=sr), result["segments"][paragraph_indices[-1]]["end"] + buffer_duration)
                end_sample = int(end_time * sr)
                start_sample = int(result["segments"][paragraph_indices[0]]["start"] * sr)
                paragraph_audio = y[start_sample:end_sample]


                # Save paragraph audio
                paragraph_filename = generate_text_based_filename(paragraph_text, paragraph_counter, paragraphs_dir_path)
                sf.write(paragraph_filename, paragraph_audio, sr)

                # --- Insert Paragraph Data ---
                paragraph_id = insert_paragraph_data(db_path, file_id, paragraph_filename, paragraph_text)

                # --- Insert Sentence Data ---
                for i in paragraph_indices:
                    segment = result["segments"][i]
                    start_time = segment["start"]
                    end_time = segment["end"]
                    sentence_text = segment["text"]
                    # Find corresponding sentence file:
                    sentence_filename = ""
                    for fname in os.listdir(sentences_dir_path):
                        if fname.startswith(f"{i+1:04d}_"): # Find matching sentence number
                            sentence_filename = fname
                            break

                    insert_sentence_data(db_path, paragraph_id, sentence_filename, sentence_text, start_time, end_time)
                paragraph_counter += 1

            output_data = {
                "transcription": result["text"],
                "segments": result["segments"],
                "dtmf_tones": dtmf_times,
                "paragraphs": paragraphs,
            }

            output_filename =  "transcription.json"
            output_path = os.path.join(output_dir_path, output_filename)
            with open(output_path, "w") as outfile:
                json.dump(output_data, outfile, indent=4)

            print(f"Transcription, sentence slices, and paragraph slices saved to: {output_dir_path}")

if __name__ == "__main__":
    main()