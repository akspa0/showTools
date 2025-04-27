# Project Brief

WhisperBite provides a comprehensive audio processing pipeline, accessible via a **Gradio web UI (`app.py`)** and a core script (`whisperBite.py`), focused on generating detailed, speaker-separated transcriptions from audio **and video** files.

Core goals:
1.  Accept various inputs (local audio/video files, directories [newest file], URLs) through the UI or script.
2.  **Automatically extract audio from video inputs using ffmpeg.**
3.  Identify and separate different speakers in the audio (diarization).
4.  **Optionally improve speaker separation accuracy via a second-pass analysis (`--enable_second_pass`).**
5.  Transcribe the speech for each speaker accurately using OpenAI's Whisper.
6.  Provide word-level timestamps and extract individual word audio snippets (**optional, `--enable_word_extraction`, default off**).
7.  Optionally enhance audio quality through normalization and vocal separation.
8.  **Optionally detect non-speech sounds (if vocal separation is enabled).**
9.  Produce structured output including transcripts (with formatted speaker labels `S0`, `S1`, etc., and sound events), segmented audio, and word data, accessible via download in the UI or directly in the filesystem.
10. Offer a user-friendly web interface for configuration and execution (with manual speaker count as default).

[The foundational document. Define core requirements and goals. Source of truth for project scope.] 