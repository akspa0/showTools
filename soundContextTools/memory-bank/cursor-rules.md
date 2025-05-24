# cursor-rules.md

## Project Intelligence & Patterns

- No file in the codebase may exceed 750 lines of code. If a file approaches this limit, logic must be refactored into smaller, well-named modules or utility files. This applies to all Python source files, pipeline scripts, and modules.

- All pipeline stages must be managed by a PipelineOrchestrator, which tracks job state, progress, and logging. The orchestrator must support global and per-file progress tracking (e.g., tqdm), robust error handling, and future extensibility (parallelism, job queuing, UI, etc.).

- **Fundamental privacy rule:** No logging or console output (including errors) may occur for any file until after it has been renamed and the original in raw_inputs has been deleted. All logging and manifest writing must only use anonymized fields (output_name, output_path, tuple_index, subid, type, timestamp) and must occur after the file is fully anonymized. This rule applies to all pipeline stages and the orchestrator.

- All audio file processing must remove PII from filenames at ingestion.
- File tuples are tracked by a unique, zero-padded chronological index (<0000>, <0001>, ...) throughout the pipeline, assigned based on timestamp.
- The index is used as a prefix for all output folders, files, manifest entries, and metadata fields for strict chronological tracking and traceability.
- Each processing step must maintain traceability and update filenames accordingly.
- Use modular, batch-oriented processing for scalability.
- Metadata lineage is crucial: use mutagen to read, update, and propagate ID3 tags at every step, ensuring all outputs can be traced back to their source.
- Always resample to 16kHz mono only for model inference (pyannote, parakeet); never use 16kHz files for final outputs or soundbites.
- All soundbites and final outputs must be cut from the highest quality, normalized split vocal audio available.
- Manifest must track all file versions, sample rates, indices, and processing lineage for full traceability.
- Remixing logic: instrumentals at 50% volume, stereo channels panned 60/40 for a 40% separation effect (stereo_left = 0.6*left_mix + 0.4*right_mix, stereo_right = 0.4*left_mix + 0.6*right_mix).
- All vocals used in remix/show must be normalized to -14.0 LUFS.
- Output directory structure must mirror input, with each call in its own indexed folder containing /soundbites, /call, /transcripts, /llm_outputs.
- Show output: concatenated calls (with optional tones), plus a text file listing call order, call start timestamps, and metadata.
- All output files must be included in the manifest, with robust error handling and defensive filtering for malformed data.
- Document any new patterns, challenges, or workflow preferences here as the project evolves.
- CLAP annotation is required for all out- prefixed files and other audio types where context is needed
- Use a configurable list of prompts for CLAP to search for relevant audio events (e.g., dogs barking, DTMF, ringing, yelling, etc.)
- Only accept CLAP annotations with a confidence level of 0.9 or higher
- All accepted CLAP annotations must be added to the master transcript and manifest for each call, to provide context for LLM and downstream processing
- After diarization, segment normalized vocals per speaker and store in speakers/SXX/ folders
- Each segment is named with a chronological index and a short, meaningful name (≤ 48 characters)
- For each segment, generate a transcription and save as a .txt file in the same folder
- Rename segment files to include the index and a short version of the transcription
- Manifest must record original input file, timestamp range, speaker ID, index, and transcription for each segment
- In later steps, convert all segments to MP3 and add ID3 tags with full lineage (input file, timestamps, speaker, transcription, etc.)
- LLM tasks are managed via workflow presets (JSON/YAML) that define task names, prompt templates, output files, and model parameters
- Each call runs all llm_tasks as defined, with prompts rendered from the transcript/context and outputs saved to the call's output folder
- All LLM outputs are referenced in the manifest for traceability and downstream use
- Workflow presets are extensible: users can add, remove, or modify tasks and prompts without code changes
- For long audio files, enable CLAP-driven segmentation mode via CLI/config to detect call boundaries and segment audio into individual calls
- Segmentation prompts (e.g., 'telephone ring tones', 'hang-up tones') are user-extensible via workflow config
- Each segmented call is processed as a single-file call (mono vocals), with all lineage tracked in the manifest
- All workflows are JSON files in workflows/ and define pipeline logic, routing, and configuration
- User-specific data (e.g., HuggingFace token, credentials, preferences) is stored in config/ and never mixed with workflow logic
- Use huggingface-cli to set the token, and read from config/ at runtime
- After remixing, mix vocals and instrumentals for trans_out and recv_out pairs into left and right channels, with 50% volume on instrumental track
- Produce new call files with chronological call index as filename and LLM-generated call title, sanitized of all punctuation from LLM call title response
- Optionally apply tones to call files (if not in show-mode)
- Show output: concatenate all valid calls (>10s, error-free) into a single WAV file, insert tones between calls, and include a text file listing call order, names, timestamps, and metadata
- LLM config, prompts, and output mapping are user-extensible via workflow JSONs; CLI options allow override of LLM config and call tones.
- Defensive programming and robust error handling are required at every stage; skip and warn on malformed data, and ensure all outputs are auditable and extensible.

---

### New Patterns (2024-06-XX)
- Each run creates a timestamped output folder with stage-based subfolders (raw_inputs, renamed, separated, normalized, clap, diarized, soundbites, transcription).
- Manifest for call tuples never contains original filenames, only renamed, PII-free filenames and lineage.
- Diarization segments vocal audio, which is then transcribed (Parakeet) into soundbites; each soundbite and its transcription is saved in soundbites/.
- Master transcript aligns all diarized utterances and CLAP annotations in true chronological order.
- Transcript format: [CHANNEL][SPEAKER][START-END] for utterances, [CLAP][START-END][annotation] for CLAP events.
- All entries in the master transcript are sorted by start time for context and LLM use.
- **NEW (2024-06): Soundbites folders are now named after sanitized call titles, not just call IDs.**
- **NEW: SFW (safe-for-work) call titles and show summaries are generated and included in outputs.**
- **NEW: Show notes are generated by the LLM and appended to show summaries.**
- **NEW: Extensions system: users can add scripts (like character_ai_description_builder.py) to an extensions/ folder, which run after the main pipeline and can use all outputs.**
- **NEW: CLAP annotation is now optional and disabled by default; out- files are only processed for CLAP unless --process-out-files is set.**
- **NEW: All outputs, logs, and manifests are strictly PII-free and fully auditable.**
- **NEW: Timestamps in show summaries are now in HH:MM:SS format, with emoji for calls/tones.**
- **NEW: Resume and force: The pipeline is robust to resume, with --resume and --resume-from, and --force now archives outputs instead of deleting them.**
- **Next: Build out extension API and document extension interface.**

- **Finalization stage:**
  - All valid soundbites and show audio are converted to 192kbps VBR MP3s in a finalized/ output folder.
  - ID3 tags are embedded in all MP3s, including call index, channel, speaker, segment index, timestamps, transcript, LLM-generated call title, and full lineage.
  - LLM-driven show title and description are generated via a secondary LLM task using all call titles and synopses; show title is a short, family-friendly, comedic sentence or phrase.
  - Show MP3 and .txt description are named after the sanitized LLM show title; fallback to completed-show.mp3 if LLM output is missing/invalid.
  - Show description is included in manifest, ID3 tags, and as a separate .txt file.
  - Two-stage LLM workflow: per-call titles/synopses, then show-level title/description.
  - Manifest and logs are updated with all finalized outputs and metadata.

- **All logging and manifest writing must be strictly PII-free. No original filenames, paths, or other PII may be logged or output at any stage. Logging and manifest writing are only permitted after all files are anonymized and originals are deleted, and only anonymized fields (output_name, output_path, tuple_index, subid, type, timestamp) may be used. This applies to all pipeline stages, scripts, and utilities.**

_This file is a living document. Update it as you discover new patterns or preferences in the project._ 