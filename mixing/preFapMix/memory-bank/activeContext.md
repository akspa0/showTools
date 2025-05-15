# Active Context

## Current Focus
- Refining and thoroughly testing the current audio processing pipeline in `audiotoolkit_phase1.py`.
- Validating the default loudness settings:
    - Vocal stems: `0.0 LUFS` (audiomentations)
    - Instrumental stems: `-14.0 LUFS` (audiomentations)
    - Optional vocal stem limiting: Peak at `-0.5 dBFS` (audiomentations, via `--enhance_vocals`)
    - Final mixed output: `-8.0 LUFS` (ffmpeg loudnorm)
- Ensuring robustness of the `yt-dlp` integration for URL inputs.
- Confirming correct handling of various input file types and scenarios (pairs, singles, directory input).

## Recent Changes
- **Successfully implemented a multi-stage loudness normalization strategy that has resolved previous issues with quiet audio and buried vocals.**
    - Vocal stems are normalized to `0.0 LUFS` by default using `audiomentations.LoudnessNormalization`.
    - Instrumental stems are normalized to `-14.0 LUFS` by default using `audiomentations.LoudnessNormalization`.
    - An optional limiter (`audiomentations.Limiter`) is applied to vocal stems if `--enhance_vocals` is true (default peak `-0.5 dBFS`).
    - The final mixed output (after stem processing and L/R mixing) is normalized to `-8.0 LUFS` using `ffmpeg loudnorm`.
- **User confirmed these default settings produce excellent audio quality.**
- Corrected channel mapping: `recv_out` to LEFT, `trans_out` to RIGHT.
- Implemented timestamped output run folders (e.g., `output_dir/run_YYYYMMDD_HHMMSS/`).
- Implemented directory input scanning for `trans_` and `recv_` prefixed files.
- Retained `--save_intermediate_stems` for debugging loudness at each stage.

## Next Steps
1.  Conduct comprehensive testing of `audiotoolkit_phase1.py` with a diverse set of audio inputs (various lengths, qualities, mono/stereo sources for 'single' processing, `trans_`/`recv_` pairs).
2.  Review and enhance error handling for all critical processing stages (download, separation, normalization, mixing).
3.  Begin planning for Phase 2: Context-aware transcription (`WhisperBite`) and CLAP event integration (`ClapAnnotator`), considering how the processed audio from Phase 1 will feed into these.
4.  Consider adding a configuration file for managing default parameters if CLI arguments become too numerous for frequent adjustments.

## Active Decisions
- The current multi-stage loudness strategy (Vocals `0.0 LUFS`, Instrumentals `-14.0 LUFS` via `audiomentations`; optional vocal limiting; Final Mix `-8.0 LUFS` via `ffmpeg loudnorm`) is the accepted approach for achieving desired audio balance and overall loudness.
- Default values for LUFS targets and limiter settings are based on user-confirmed optimal results.
- Continue to prioritize command-line operation for `audiotoolkit_phase1.py`.
- Phase 1 (Advanced Pre-Processing, Separation, Stem Modification & Mixing) is largely complete in terms of core audio manipulation.

## Recent Changes
- Refined `call_id` generation to `call_YYYYMMDD-HHMMSS`.
- Implemented workaround for `audio-separator`'s `--custom_output_names` by using a temporary input file and renaming its default outputs.
- Successfully fixed `audio-separator` stem file discovery.
- **Implemented directory input scanning in `audiotoolkit_phase1.py`**:
    *   Modified CLI args to accept an optional `--input_dir`.
    *   Script now scans the directory for `trans_` or `recv_` prefixed audio files.
    *   Combines these with explicitly provided file/URL inputs.

## Next Steps
1.  **Implement critical channel mapping correction:**
    *   Modify `identify_channel_pairs_and_singles` to assign `recv_out` as `left_src` and `trans_out` as `right_src`.
    *   Ensure `process_calls_phase1` uses these corrected assignments.
2.  **Implement timestamped output folders:**
    *   Modify `process_calls_phase1` to create a unique run folder (e.g., `run_YYYYMMDD_HHMMSS`) inside the main `--output_dir` for each execution.
3.  **Implement optional vocal enhancement:**
    *   Add CLI flag (e.g., `--enhance_vocals`, default True).
    *   Create `enhance_vocals` function using FFmpeg (high-pass, soft limiter).
    *   Integrate into the vocal stem processing workflow.
4.  Thoroughly test all new functionalities and corrections.
5.  Proceed to Phase 2: Context-aware transcription and CLAP event integration.

## Active Decisions
- Correct channel mapping is `recv_out` -> LEFT channel, `trans_out` -> RIGHT channel.
- Timestamped output folders are necessary for iterative testing and run management.
- Vocal enhancement (high-pass, soft limiter to -6dB) should be an option to improve speech clarity, especially after reducing instrumental volume.
- Focusing on a streamlined approach for Phase 1 (Pre-Processing, Separation, Stem Modification & Mixing).
- Using `audio-separator` CLI for stem separation.
- Prioritizing `trans_` and `recv_` file pairing from directory inputs.
- Maintaining PII safety with `call_id` in temporary filenames.

## Recent Changes
- Refined the project scope to focus on essential functionality
- Updated memory bank to reflect the streamlined approach
- Identified key components from v3.8 projects that can be integrated
- Developed a plan for a simplified but complete processing pipeline

## Next Steps
1. Implement file_matcher.py for matching recv_out/trans_out pairs
2. Integrate python-audio-separator for vocal/instrumental separation
3. Implement direct OpenAI Whisper integration to replace fap
4. Create audio_processor.py for normalization, resampling, and mixing
5. Implement show_builder.py for show file and metadata generation
6. Add controls for adjusting instrumental volume in the final mix
7. Update UI to expose essential options
8. Create comprehensive testing for the streamlined pipeline

## Active Decisions
- Focusing on a streamlined approach that covers essential functionality
- Using direct file matching for recv_out/trans_out pairs rather than complex solutions
- Replacing fish-audio-processor with OpenAI Whisper for better transcription quality
- Using python-audio-separator for vocal/instrumental separation
- Using transcript content directly for call identification
- Organizing output files chronologically for easier analysis
- Creating consolidated show files with detailed metadata
- Adding adjustable instrumental volume to improve clarity
- Maintaining a modular architecture to allow for future enhancements 