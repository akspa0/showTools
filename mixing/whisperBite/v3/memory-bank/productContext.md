# Product Context

**Problem:** Analyzing spoken audio, especially conversations with multiple speakers, is time-consuming. **Getting audio from video files requires extra steps.** Manually transcribing, identifying who said what, and finding specific words is inefficient. Furthermore, automated speaker diarization isn't always perfect and can sometimes group speech from multiple speakers into a single segment.

**Solution:** WhisperBite automates this process via a user-friendly web interface (`app.py`) or a command-line tool (`whisperBite.py`). It accepts **audio or video files**, URLs, or folders (processing the newest compatible file within). **For video inputs, it automatically extracts the audio** before performing diarization and transcription.
*   **Refinement:** It includes an **optional second pass** that re-analyzes longer segments from the first pass to detect and separate missed speaker changes, improving accuracy.
*   **Word Extraction:** It can optionally isolate individual words with their timings and audio snippets, though this feature is **disabled by default** as it generates many files.

**How it Works (User Perspective - Web UI):**
1.  Launch the Gradio app (`python app.py`).
2.  Provide source (upload audio/video file, specify folder path [processes newest file], or enter URL).
3.  Configure options: Whisper model, speaker settings (initial estimate or auto-detect), vocal separation.
4.  **Configure advanced options:** Enable second pass refinement? Enable word audio extraction?
5.  Provide Hugging Face token (for diarization).
6.  Specify output directory.
7.  Click "Process Audio". (The app extracts audio from video automatically if needed).
8.  Status updates appear. If second pass is enabled, this stage will take longer.
9.  Upon completion, a transcript preview (preferring the refined transcript if second pass was run) is shown, and a download link for the results zip file is provided.

**User Experience Goals:**
*   Provide an intuitive web UI for easy configuration and execution.
*   **Support direct processing of video files by handling audio extraction.**
*   Offer a fallback command-line interface for advanced use cases.
*   Automate complex audio processing tasks.
*   **Address common diarization inaccuracies with an optional refinement step.**
*   Generate detailed and usable outputs.
*   **Allow users to control the verbosity of output (e.g., word extraction).**
*   Offer flexibility through optional processing steps and model choices. 