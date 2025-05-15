# Progress Tracking

## What Works
- Basic audio processing pipeline structure in `audiotoolkit_phase1.py`.
- Input via command-line arguments (individual files/URLs and `--input_dir` for `trans_`/`recv_` files).
- Call ID generation (`call_YYYYMMDD-HHMMSS`).
- Downloading audio from URLs (`yt-dlp`).
- Audio stem separation using `audio-separator` CLI.
- **Effective multi-stage loudness management (user-confirmed good results):**
    - **Vocal stems normalized to `0.0 LUFS` (default) using `audiomentations.LoudnessNormalization`.**
    - **Instrumental stems normalized to `-14.0 LUFS` (default) using `audiomentations.LoudnessNormalization`.**
    - **Optional vocal stem peak limiting (e.g., to `-0.5 dBFS`) using `audiomentations.Limiter` (via `--enhance_vocals`).**
    - **Volume adjustment for stems (e.g., `--volume_vocals`, `--volume_instrumental`) applied after normalization.**
    - **Final mixed output normalized to `-8.0 LUFS` (default) using `ffmpeg loudnorm`.**
- Summing of stems for each source into a mono track (using FFmpeg).
- Stereo mixing for paired `trans_out` (Right) / `recv_out` (Left) sources with 60/40 panning (corrected mapping).
- Appending `tones.wav` to calls (configurable).
- Temporary file management within `temp_processing/<call_id>/`.
- Timestamped output run folders (e.g., `run_YYYYMMDD_HHMMSS`) in the output directory.
- Logging setup with configurable levels.
- `--save_intermediate_stems` for detailed debugging of audio stages.

## What's Left to Build

### Phase 1 (Advanced Pre-Processing, Separation, Stem Modification & Mixing)
- Comprehensive testing with diverse audio inputs.
- Enhanced error handling and reporting for all critical processing stages.
- (Potentially) Configuration file for default settings if CLI args become unwieldy.

### Phase 2 (Context-Aware Transcription & CLAP Events)
- Integration of `ClapAnnotator` logic for CLAP event detection.
  - Run CLAP detection on both vocal and instrumental stems *before* transcription.
- Context-aware transcription using `WhisperBite` mechanics, enriched by CLAP events.
- Intelligent filename generation based on transcription and context.

### Phase 3 (Synopsis, Show Compilation, LLM Processing)
- Synopsis generation.
- Show compilation with timestamped "calls" and `tones.wav` markers.
- LLM processing for insights.
- Soundbite extraction.

### General/Future
- GUI interface (re-integration or new build).
- More sophisticated `call_id` generation if needed.
- Option to use different separator models.
- Configuration file for default settings.
- More advanced audio processing options (e.g., noise reduction).
- Full integration of `pyannote.audio` for diarization with HF token handling.

## Current Status (Post User Feedback on Loudness Fixes)
- **Phase 1: Core audio processing is largely complete and producing good quality output, especially regarding loudness and vocal/instrumental balance, per user confirmation.**
- Key functionalities implemented: input handling (files, URLs, dir), stem separation, multi-stage loudness normalization (stems via audiomentations, final mix via ffmpeg loudnorm), stem volume adjustments, stereo mixing with correct channel mapping, timestamped outputs, and intermediate stem saving.
- The critical issues of quiet audio and buried vocals have been resolved with the new default LUFS targets (`0.0` for vocals, `-14.0` for instrumentals, `-8.0` for final mix).

## Known Issues
- Comprehensive testing across a wider variety of audio files is still needed to ensure robustness.
- Error handling, while present, could be further improved for edge cases.
- No specific vocal enhancement/EQ stage yet. (Note: Limiter is present, but specific EQ was removed previously, current focus is on loudness balance)
- No GUI yet.

## Current Status
- Memory bank updated with refined, focused approach
- Core architecture design completed
- Implementation plan established for essential functionality
- Components from v3.8 projects identified for integration

## Known Issues
- Current implementation doesn't match recv_out/trans_out pairs
- No vocal/instrumental separation
- Relies on fish-audio-processor for transcription
- No mechanism for generating consolidated show files
- No metadata for call timestamps
- No mechanism for adjusting instrumental volume in the final mix 