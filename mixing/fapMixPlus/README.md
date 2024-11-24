## fapMixPlus (WebUI)

### Overview
The `fapMixPlus (WebUI)` project combines the power of the `fapMixPlus.py` audio processing script with an intuitive Gradio-based user interface. This tool automates and streamlines audio downloading, conversion, separation, slicing, transcription, and file organization. Whether you’re working with uploaded files or audio from a URL, this application provides an efficient and user-friendly way to process audio files.

---

### Features
- **Audio Downloading**: Fetch audio from URLs (e.g., YouTube).
- **File Upload Support**: Process locally stored audio files.
- **WAV Conversion**: Convert input audio to the WAV format.
- **Audio Separation**: Isolate vocal tracks from WAV files.
- **Slicing and Transcription**: Segment audio and generate transcriptions for each slice.
- **File Renaming and Organization**: Sanitize filenames and structure output directories.
- **ZIP File Generation**: Compress final results for easy distribution.
- **Real-Time Logs and Controls**: Monitor progress and manage processes through a web interface.

---

### Prerequisites
1. **Python 3.7+**
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. **Fish Audio Preprocessor (`fap`)**:
   - Clone the [Fish Audio Preprocessor repository](https://github.com/fishaudio/audio-preprocess):
     ```bash
     git clone https://github.com/fishaudio/audio-preprocess.git
     ```
   - Navigate to the repository and install:
     ```bash
     cd audio-preprocess
     pip install -e .
     ```
   - Verify the installation:
     ```bash
     fap --version
     ```

---

### Usage

#### Running the WebUI
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/fapMixPlus-webui.git
   cd fapMixPlus-webui
   ```
2. Start the Gradio-based application:
   ```bash
   python app.py
   ```
3. Access the WebUI in your browser (URL is provided in the console).

#### WebUI Options
- **Audio URL**: Enter a YouTube or supported audio link to download and process.
- **Upload Files**: Upload audio files (`.wav`, `.mp3`) for local processing.
- **Output Directory**: Specify a directory for storing output files.
- **Buttons**:
  - `Process Audio`: Start the audio processing.
  - `Cancel Process`: Stop any running process.
  - `Download ZIP`: Download the latest `.zip` archive.
  - `Refresh Logs`: View updated logs in real time.

---

### CLI Usage (Optional)
You can also use `fapMixPlus.py` directly as a standalone command-line tool:
```bash
python fapMixPlus.py --url https://youtu.be/example_video --output_dir my_output
```

---

### Output Structure
Processed files are organized into structured directories:
```
output_<timestamp>/
├── wav_conversion/            # WAV-converted audio files
├── separation_output/         # Separated vocal track files
├── slicing_output/            # Sliced audio segments
├── final_output/              # Renamed and sanitized WAV and .lab files
└── zip_files/                 # Compressed archives of final output
```

---

### Example Workflow
1. **Audio Input**: Upload files or provide a URL.
2. **Processing Stages**:
   - Convert audio to WAV format.
   - Separate vocals from the WAV files.
   - Slice and transcribe vocal tracks.
   - Rename and organize transcription files.
3. **Output**:
   - Access the organized files in `final_output/`.
   - Download the `.zip` archive from the WebUI or `zip_files/`.

---

### Notes
- Ensure the `fapMixPlus.py` script and required dependencies (`fap`) are properly installed.
- Final `.wav` and `.lab` filenames are sanitized to include the transcription content or a numerical prefix.

---

### Example Final Output Filenames
- `0001_Hello_this_is_a_sample.wav`
- `0001_Hello_this_is_a_sample.lab`

Files without valid transcription will retain only the numerical prefix:
- `0002.wav`
- `0002.lab`

---

### License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

### Acknowledgments
- Gradio for the interactive web interface.
- `fapMixPlus.py` for backend audio processing.
- `yt-dlp` for YouTube audio downloading capabilities.