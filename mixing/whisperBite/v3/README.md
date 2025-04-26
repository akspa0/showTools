# WhisperBite

WhisperBite is a tool built with Gradio that processes audio and video files to perform speaker diarization (identifying who spoke when) and transcription using OpenAI's Whisper model. It can handle various inputs, separate vocals, extract individual words, and refine results with a second pass.

## Features

*   **Input:** Accepts single audio/video files, folders (processes newest file), or URLs (YouTube, direct links).
*   **Audio Extraction:** Automatically extracts audio from video inputs.
*   **Normalization:** Normalizes audio loudness to a standard level.
*   **Vocal Separation (Optional):** Uses Demucs to separate vocals from background noise/music.
*   **Speaker Diarization:** Identifies different speakers using `pyannote.audio`.
*   **Transcription:** Transcribes the speech for each identified speaker using Whisper.
*   **Output:** Creates:
    *   Individual audio segments per speaker turn.
    *   Text transcripts per speaker turn.
    *   A master transcript combining all speakers chronologically.
    *   Optional individual word audio snippets with timestamps.
    *   A zip file containing all results.
*   **Second Pass Refinement (Optional):** Re-analyzes longer segments to potentially improve speaker separation accuracy.

## Prerequisites

1.  **Python:** Version 3.9 or higher recommended.
2.  **ffmpeg:** Required for audio normalization and extraction from video. You must install it separately and ensure it's available in your system's PATH.
    *   **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin` directory to your PATH environment variable.
    *   **macOS (using Homebrew):** `brew install ffmpeg`
    *   **Linux (using apt):** `sudo apt update && sudo apt install ffmpeg`
3.  **PyTorch:** The `requirements.txt` file lists `torch`. Depending on your system (CPU-only or GPU with CUDA), you might need a specific version. Check the [PyTorch website](https://pytorch.org/get-started/locally/) for the correct installation command for your setup.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url> # Replace with the actual URL
    cd whisperBite
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you encounter issues with PyTorch, install it manually first using the command from the PyTorch website, then run the command above again.*

4.  **Hugging Face Token:**
    *   `pyannote.audio` requires a Hugging Face token for accessing diarization models.
    *   Get a token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (read access is sufficient).
    *   You can either:
        *   Set the `HF_TOKEN` environment variable before running the app.
        *   Enter the token directly into the "Hugging Face Token" field in the Gradio UI.

## Running the Application

Once setup is complete, run the Gradio application:

```bash
python app.py
```

This will start a local web server. Open the provided URL (usually `http://127.0.0.1:7860`) in your browser to use the interface.

## Notes

*   **Vocal Separation:** Requires `demucs` to be installed (`pip install demucs`). If enabled, separation is performed once on the normalized audio *before* the first diarization pass. The second pass refinement (if enabled) uses segments derived from this initially separated audio.
*   **Second Pass:** This feature analyzes segments longer than 5 seconds from the first pass to try and refine speaker labels. It can be time-consuming.
*   **Resource Usage:** Whisper models (especially larger ones) and Demucs can be computationally intensive and require significant RAM/VRAM.

## License

MIT
