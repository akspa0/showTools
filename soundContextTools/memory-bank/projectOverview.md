# Project Overview

## Purpose
A modular, extensible audio context tool for processing phone calls and other audio files through advanced AI models. Automates context annotation, diarization, normalization, remixing, transcription, and soundbite extraction, with robust file tracking, privacy, and full data lineage.

---

## Core Features & Workflow

1. **Input Handling**
   - Supports batch processing of folders (with nested subfolders) containing phone call tuples (`out-`, `trans_out-`, `recv_out-`) and other audio files.
   - Recursive directory scanning and mirrored output structure.
   - PII removal and renaming with a unique, zero-padded chronological index (e.g., `<0000>`), based on timestamps.
   - Skips files under 9 seconds (calls for show output must be >10s and error-free).

2. **Audio Processing Pipeline**
   - **Separation:** Uses `audio-separator` to split `trans_out-` (Right) and `recv_out-` (Left) into vocals/instrumentals.
   - **Normalization:** Applies loudness normalization (audiomentations) to vocals.
   - **Remixing:** Combines vocals/instrumentals per channel (instrumentals at 50% volume), then creates a stereo file with 20% channel separation (fake-stereo effect).
   - **Resampling:** All model inference (pyannote, parakeet) uses 16kHz mono; all outputs/soundbites use highest quality (44.1kHz stereo).
   - **Metadata:** Uses mutagen to propagate and update ID3 tags and preserve original timestamps for all outputs.

3. **Context Annotation (CLAP)**
   - Runs on `out-` files and other audio as configured.
   - Prompts are workflow-driven and user-extensible (JSON config).
   - Confidence threshold (default 0.6) for accepted annotations.
   - CLAP can also segment long audio into calls (e.g., by detecting ring/hang-up tones).
   - All accepted annotations are merged into the master transcript and manifest.

4. **Diarization & Transcription**
   - Diarization (pyannote) on highest quality vocals, not 16kHz input.
   - Segments audio per speaker, stores in `speakers/SXX/` folders.
   - Each segment is named with a chronological index and a short, meaningful name (≤ 48 chars).
   - Transcription (parakeet) for each segment, with .txt output in the same folder.
   - Manifest records all lineage: input file, timestamps, speaker, index, transcription.
   - Later: Convert segments to MP3 with full ID3 tags and lineage.

5. **LLM Integration**
   - Workflow-driven LLM tasks (JSON in `workflows/`), e.g., call title, synopsis, categories, image prompt, song, etc.
   - Uses local LLM (e.g., via lmstudio) with configurable prompts and output mapping.
   - All LLM outputs are saved per call and referenced in the manifest.

6. **Call mixing**
   - Mixes Vocals and Instrumental trans_out and recv_out pairs back into left and right channels, with 50% volume on instrumental track
   - Produces new Call files with chronological call index as filename and llm-generated call title, sanitized of all punctuation from llm call title response.
   - Optionally applies tones to call files (if not in show-mode)

7. **Show Output**
   - Concatenates all valid calls (>10s, error-free) into a single WAV file, inserting a tone (`tones.wav`) between calls.
   - Includes a text file listing call order, names, timestamps, and metadata.
   - Later: MP3 show output with ID3 tags.

8. **Extensibility & User Config**
   - All workflows are JSON files in `workflows/` (routing, prompts, tasks, etc.).
   - User-specific data (e.g., HuggingFace token) is stored in `config/`.
   - CLI/config options for segmentation, prompt management, and more.

---

## Technical Stack
- Python 3.x
- parakeet (HuggingFace)
- audio-separator
- CLAP (HuggingFace)
- audiomentations
- mutagen
- pyannote (planned)
- lmstudio/OpenAI (for LLM tasks)

---

## Design Patterns & Best Practices
- Modular, batch-oriented pipeline
- Manifest and metadata for full traceability
- Workflow/config separation for user flexibility
- All outputs and temp files preserve original timestamps
- Robust error handling and skip logic for show output
- User-extensible prompts and LLM tasks

---

## Folders & Structure
- `workflows/` — JSON workflow presets (routing, prompts, LLM tasks)
- `config/` — User-specific data (tokens, preferences)
- `tones.wav` — Default tone for show output (user-replaceable)
- Output: Mirrored input structure, with per-call folders and subfolders for soundbites, call audio, transcripts, LLM outputs, and show

---

## Outstanding/Planned
- Automated tests and validation
- Developer/contributor documentation
- Advanced analytics, UI, and deployment (future)
- Security/privacy enhancements (e.g., secure temp file deletion)

---

_Edit this file to add, clarify, or expand on any aspect of the project!_ 