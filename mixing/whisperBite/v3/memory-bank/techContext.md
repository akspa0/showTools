# Tech Context

**Core Language:** Python 3

**Key Libraries:**
*   **`gradio`**: Used by `app.py` to create the web UI.
*   `openai-whisper`: Core speech-to-text transcription (`whisperBite.py`).
*   `pyannote.audio`: Speaker diarization (`whisperBite.py`). Requires Hugging Face token.
*   `torch`: Required backend for Whisper and Pyannote (supports CPU/GPU).
*   `pydub`: Audio manipulation (`whisperBite.py`).
*   `argparse`: Command-line argument parsing.
*   `requests`: For downloading URLs (via `utils.py`).
*   Standard libraries: `logging`, `os`, `sys`, `datetime`, `json`, `subprocess`, `shutil`, `glob`, `tempfile`.

**External Dependencies (Command Line Tools):**
*   `ffmpeg`: **Required** for audio normalization and **video audio extraction**.
*   `demucs`: Optional dependency for vocal separation.

**Development Setup:**
*   Python environment with libraries installed (e.g., via `pip`).
*   `ffmpeg` and optionally `demucs` installed and in PATH.
*   Internet access for model/audio downloads.
*   Hugging Face account and API token required for diarization.

**Execution:**
*   **Web UI:** `python app.py` (Optional args: `--public`, `--port`).
*   **CLI:** `python whisperBite.py`.

**Hardware:**
*   CUDA-compatible GPU recommended, CPU supported.

**Configuration:**
*   **Web UI (`app.py`):** Interactive Gradio components.
*   **CLI (`whisperBite.py`):** `argparse` arguments, including:
    *   `--enable_word_extraction`: Toggle word audio snippet generation (default: off).
    *   `--enable_second_pass`: Toggle diarization refinement pass (default: off).
*   Some internal parameters in `whisperBite.py` are hardcoded (LUFS target, segment merge gap, word padding, second pass thresholds). 