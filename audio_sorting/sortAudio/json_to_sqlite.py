import argparse
import json
import os
import sqlite3
import datetime

def create_database(db_path):
    """Creates the SQLite database and tables."""
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
    #  speakers and speaker_segments tables are created in diarize_audio.py, and are not needed here

    conn.commit()
    conn.close()

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
    parser = argparse.ArgumentParser(description="Convert JSON output from audio_processor.py to SQLite database.")
    parser.add_argument("input_dir", help="Path to the output directory of audio_processor.py")
    parser.add_argument("--db_path", default="transcriptions.db", help="Path to the SQLite database file")
    args = parser.parse_args()
    input_dir = args.input_dir
    db_path = args.db_path

    create_database(db_path)

    for output_dir in os.listdir(input_dir):
        output_dir_path = os.path.join(input_dir, output_dir)
        if os.path.isdir(output_dir_path):
            json_path = os.path.join(output_dir_path, "transcription.json")
            if os.path.exists(json_path):
                print(f"Processing: {json_path}")
                with open(json_path, "r") as f:
                    data = json.load(f)

                # Extract data from the JSON structure
                # Use the directory name as the original filename, since old JSON doesn't have it
                original_filename = output_dir
                original_filepath = os.path.join(input_dir, original_filename) # Best guess
                # Get the date from the output_dir name, if possible, otherwise use current time.
                try:
                    date_part = output_dir.split("_")[-2] # format YYYYMMDD
                    time_part = output_dir.split("_")[-1] # format HHMMSS
                    processed_date = datetime.datetime.strptime(f"{date_part} {time_part}", "%Y%m%d %H%M%S").isoformat()
                except:
                    processed_date = datetime.datetime.now().isoformat()


                file_id = insert_file_data(db_path, original_filename, original_filepath, processed_date)

                # Process paragraphs and sentences
                if "paragraphs" in data and isinstance(data["paragraphs"], list):
                    for i, paragraph_indices in enumerate(data["paragraphs"]):
                        paragraph_text = ""
                        for sentence_index in paragraph_indices:
                            if isinstance(data["segments"], list) and len(data["segments"]) > sentence_index:
                                paragraph_text += data["segments"][sentence_index]["text"] + " "
                        
                        # Construct paragraph filename based on index
                        paragraph_filename = f"{i+1:04d}.wav" #This will not match the actual filename, but we need something

                        paragraph_id = insert_paragraph_data(db_path, file_id, paragraph_filename, paragraph_text.strip())

                        for j in paragraph_indices:
                            if isinstance(data["segments"], list) and len(data["segments"]) > j:
                                segment = data["segments"][j]
                                sentence_text = segment["text"]
                                start_time = segment["start"]
                                end_time = segment["end"]
                                # Construct sentence filename based on index within paragraph
                                sentence_filename = f"{j+1:04d}.wav" #This will not match

                                insert_sentence_data(db_path, paragraph_id, sentence_filename, sentence_text, start_time, end_time)
            else:
                print(f"Skipping: {output_dir_path} (no transcription.json found)")

if __name__ == "__main__":
    main()