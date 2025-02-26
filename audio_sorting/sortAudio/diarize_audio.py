import argparse
import os
import librosa
import soundfile as sf
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import whisper
import json
import re
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

def get_paragraph_id(db_path, paragraph_filename):
    """Retrieves the paragraph_id from the database based on the filename."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT paragraph_id FROM paragraphs WHERE paragraph_filename = ?", (paragraph_filename,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0]
    else:
        return None

def create_speaker_tables(db_path):
    """Creates the speakers and speaker_segments tables if they don't exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS speakers (
            speaker_id INTEGER PRIMARY KEY,
            speaker_name TEXT UNIQUE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS speaker_segments (
            segment_id INTEGER PRIMARY KEY,
            paragraph_id INTEGER,
            speaker_id INTEGER,
            segment_filename TEXT,
            segment_text TEXT,
            start_time REAL,
            end_time REAL,
            FOREIGN KEY (paragraph_id) REFERENCES paragraphs(paragraph_id),
            FOREIGN KEY (speaker_id) REFERENCES speakers(speaker_id)
        )
    """)
    conn.commit()
    conn.close()

def create_temp_speaker_tables(db_path):
    """Creates temporary speaker tables for the diarization-only case."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    #  Create tables without foreign key constraints
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS temp_speakers (
            speaker_id INTEGER PRIMARY KEY,
            speaker_name TEXT UNIQUE
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS temp_speaker_segments (
            segment_id INTEGER PRIMARY KEY,
            paragraph_filename TEXT,
            speaker_id INTEGER,
            segment_filename TEXT,
            segment_text TEXT,
            start_time REAL,
            end_time REAL,
            FOREIGN KEY (speaker_id) REFERENCES temp_speakers(speaker_id)
        )
    """)
    conn.commit()
    conn.close()

def insert_speaker(db_path, speaker_name, temp_db=False):
    """Inserts a speaker into the database and returns the speaker_id."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    table_name = "temp_speakers" if temp_db else "speakers"
    try:
        cursor.execute(f"INSERT INTO {table_name} (speaker_name) VALUES (?)", (speaker_name,))
        speaker_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        cursor.execute(f"SELECT speaker_id FROM {table_name} WHERE speaker_name = ?", (speaker_name,))
        speaker_id = cursor.fetchone()[0]
    conn.commit()
    conn.close()
    return speaker_id

def insert_speaker_segment(db_path, paragraph_id, speaker_id, segment_filename, segment_text, start_time, end_time, temp_db=False):
    """Inserts a speaker segment into the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    if temp_db:
        cursor.execute("""
            INSERT INTO temp_speaker_segments (paragraph_filename, speaker_id, segment_filename, segment_text, start_time, end_time)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (paragraph_id, speaker_id, segment_filename, segment_text, start_time, end_time)) # paragraph_id is now filename
    else:
        cursor.execute("""
            INSERT INTO speaker_segments (paragraph_id, speaker_id, segment_filename, segment_text, start_time, end_time)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (paragraph_id, speaker_id, segment_filename, segment_text, start_time, end_time))
    conn.commit()
    conn.close()

def main():
    parser = argparse.ArgumentParser(description="Speaker Diarization and Re-transcription")
    parser.add_argument("input_dir", help="Path to the output directory of audio_processor.py")
    parser.add_argument("--model", default="turbo", help="Whisper model to use for re-transcription")
    parser.add_argument("--db_path", default="transcriptions.db", help="Path to the SQLite database file")
    args = parser.parse_args()
    input_dir = args.input_dir
    model_name = args.model
    db_path = args.db_path # Use the provided db_path

    paragraphs_dir = os.path.join(input_dir, "paragraphs")
    speakers_dir = os.path.join(input_dir, "speakers") # Output in original directory
    os.makedirs(speakers_dir, exist_ok=True)

    if not os.path.isdir(paragraphs_dir):
        print(f"Error: 'paragraphs' directory not found in {input_dir}")
        return

    # Initialize pyannote pipeline (requires HF_TOKEN in environment)
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=True)
    except Exception as e:
        print(f"Error initializing pyannote pipeline: {e}")
        print("Make sure you have an HF_TOKEN set in your environment.")
        return

    # Load Whisper model
    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)

    # --- Database Handling ---
    temp_db = False
    if not os.path.exists(db_path):
        print(f"Warning: Database file '{db_path}' not found.  Using temporary database.")
        temp_db = True
        db_path = os.path.join(input_dir, "diarization_only.db") # Use a temporary DB in the output dir
        create_temp_speaker_tables(db_path)
    else:
        # Check if the 'paragraphs' table exists
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='paragraphs'")
        table_exists = cursor.fetchone()
        conn.close()
        if not table_exists:
            print(f"Warning: 'paragraphs' table not found in '{db_path}'. Using temporary database.")
            temp_db = True
            db_path = os.path.join(input_dir, "diarization_only.db")
            create_temp_speaker_tables(db_path)
        else:
            create_speaker_tables(db_path) # Ensure tables exist


    for filename in os.listdir(paragraphs_dir):
        if filename.endswith(".wav"):
            paragraph_audio_path = os.path.join(paragraphs_dir, filename)
            print(f"Diarizing and re-transcribing: {filename}")

            # Run diarization
            with ProgressHook() as hook:
              diarization = pipeline(paragraph_audio_path, num_speakers=2, hook=hook)

            # Create speaker directories
            speaker_data = {} # Store data for this paragraph.
            for speaker_num in range(2):  # Assuming max 2 speakers
                speaker_dir = os.path.join(speakers_dir, os.path.splitext(filename)[0], f"speaker_{speaker_num}")
                os.makedirs(speaker_dir, exist_ok=True)
                speaker_data[f"SPEAKER_{speaker_num:02d}"] = {
                    "segments": [], #Keep this to simplify logic
                }

            # Get paragraph_id for database insertion (only if not using temp db)
            paragraph_id = None if temp_db else get_paragraph_id(db_path, filename)
            if paragraph_id is None and not temp_db:
                print(f"Error: Could not find paragraph_id for {filename}")
                continue # Skip this file

            # Slice, save, and re-transcribe audio segments for each speaker
            y, sr = librosa.load(paragraph_audio_path)
            segment_counter = 0

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start_time = turn.start
                end_time = turn.end
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                speaker_audio = y[start_sample:end_sample]

                speaker_dir_data = speaker_data.get(speaker)
                if speaker_dir_data:
                    # Re-transcribe
                    result = model.transcribe(speaker_audio, verbose=False) # Pass audio directly
                    segment_text = result["text"]

                    # Generate filename from transcription
                    if temp_db:
                        segment_filename = generate_text_based_filename(segment_text, segment_counter, speakers_dir)
                    else:
                        segment_filename = generate_text_based_filename(segment_text, segment_counter, speaker_dir_data["dir"])

                    segment_filepath = os.path.join(speakers_dir if temp_db else speaker_dir_data["dir"], segment_filename)

                    sf.write(segment_filepath, speaker_audio, sr) #Save the audio

                    # --- Database operations ---
                    speaker_id = insert_speaker(db_path, speaker, temp_db=temp_db) # Insert speaker and get ID
                    insert_speaker_segment(db_path, filename if temp_db else paragraph_id, speaker_id, segment_filename, segment_text, start_time, end_time, temp_db=temp_db)

                    segment_counter += 1

    print(f"Diarization and re-transcription complete. Results saved to database: {db_path}")

if __name__ == "__main__":
    main()