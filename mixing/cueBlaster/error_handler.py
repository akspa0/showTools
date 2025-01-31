def handle_missing_file(file_path):
    print(f"Error: The file {file_path} does not exist.")
    print("Please ensure all files are present and correctly named.")

def handle_mismatched_files(cue_file, audio_files):
    print(f"Error: No matching audio file found for cue file {cue_file}.")
    print("Please ensure the cue file matches an audio file in the same directory.")

def handle_encoding_error(file_path, tried_encodings):
    print(f"Error: Could not decode file {file_path}")
    print(f"Tried the following encodings: {', '.join(tried_encodings)}")
    print("The file might be using a different character encoding.")

def handle_processing_error(file_path, error):
    print(f"Error processing {file_path}: {str(error)}")