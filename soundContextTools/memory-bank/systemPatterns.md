# systemPatterns.md

**Purpose:**
Documents system architecture, key technical decisions, design patterns, and component relationships.

## Architecture Overview

- Batch-oriented pipeline for audio file ingestion, renaming, and processing
- Modular steps for PII removal, separation, annotation, normalization, diarization, metadata propagation, remixing, and show creation
- Explicit handling of sample rates, channel assignments, and recursive folder processing throughout the pipeline
- Strict chronological tracking of calls and tuples using a unique, zero-padded index (e.g., <0000>, <0001>, ...)

## Key Technical Decisions

- Assign a unique, zero-padded chronological index to each call tuple, based on timestamp
- Use the index as a prefix for all output folders, files, and manifest entries
- Remove PII from all filenames at ingestion
- Maintain traceability by prefixing all outputs with tuple index
- Use open-source, state-of-the-art models for each processing step
- Propagate and update audio metadata (ID3 tags) at every step using mutagen
- All input and output files are 44.1kHz stereo; resample to 16kHz mono only for model inference (pyannote, parakeet)
- Soundbites and final outputs are always cut from the highest quality, normalized split vocal audio (not from 16kHz model input)
- Remixing: vocals and instrumentals are mixed per channel (instrumentals at 50% volume), then channels are combined into a stereo file with 20% separation per channel (40% total center separation)
- Output directory structure mirrors input, with each call in its own indexed folder containing /soundbites, /call, /transcripts, /llm_outputs
- Show output: concatenate all valid calls (>10s, error-free) into a single WAV file, insert tones between calls, and include a text file listing call order, names, timestamps, and metadata
- Integrate CLAP annotation as a core step for all out- prefixed files and other audio types
- Use a configurable set of prompts for CLAP to detect contextual audio events (e.g., dogs barking, DTMF tones, ringing, yelling, etc.)
- Set a confidence threshold of 0.6 for accepting CLAP annotations
- Merge CLAP annotations into the master transcript and manifest for each call, providing context for downstream LLM processing
- Diarization (pyannote) is performed on the highest quality, normalized vocals (not 16kHz model input)
- Diarization output is used to segment audio into per-speaker, per-segment files
- Each speaker's segments are stored in a folder structure: speakers/S00/, speakers/S01/, etc.
- Each segment file is named with a chronological index and a short, meaningful name (≤ 488 characters)
- For each segment, a transcription is generated (parakeet) and saved as a .txt file in the same folder
- Segment files are renamed to include the index and a short version of the transcription
- Manifest records original input file, timestamp range, speaker ID, index, and transcription for each segment
- In later steps, all segments are converted to MP3 and tagged with full lineage (input file, timestamps, speaker, transcription, etc.)
- LLM task management is driven by configurable workflow presets (JSON/YAML) that define llm_tasks, model parameters, and output mapping
- Each llm_task specifies a name, prompt_template (with placeholders like {transcript}), and output_file
- The LLM module (e.g., llm_module.py) executes all tasks per call, saving outputs to the call's output folder and returning output paths for manifest integration
- Workflow presets are fully extensible: users can add, remove, or modify tasks and prompts without code changes
- For long audio files (e.g., ≥10 minutes), enable CLAP-driven segmentation mode via CLI/config
- Use CLAP prompts (e.g., 'telephone ring tones', 'hang-up tones') to detect call boundaries and segment audio into individual calls
- Each segment is processed as a single-file call (mono vocals, no left/right distinction), following the standard call pipeline
- CLI/config allows users to enable/disable segmentation mode and specify segmentation prompts
- Manifest records original file, segment boundaries, and full processing lineage for each segmented call
- This segmentation approach is extensible to other use cases (e.g., segmenting by other sound events)
- All workflows are defined as JSON files and stored in a workflows/ folder at the project root
- Each workflow JSON contains routing/configuration for audio processing (LLM tasks, CLAP prompts, segmentation, etc.)
- A separate config/ folder stores user-specific data, such as HuggingFace token (set via huggingface-cli) and other credentials or preferences
- The program reads from config/ for user data and from workflows/ for pipeline logic

## Design Patterns

- File tuple identification: out-, trans_out-, recv_out- prefix matching
- Filename parsing and reformatting to <0000>-<prefix>-YYYYMMDD-HHMMSS.wav
- Output file/folder naming: always prefix with chronological index
- Modular pipeline: each step operates on tracked files, outputs to next step
- Metadata pattern: At each step, read, update, and write metadata to preserve lineage and context
- Resampling pattern: Only resample to 16kHz for model input; all other processing uses highest available quality
- Manifest pattern: Track all file versions, sample rates, indices, and processing lineage
- Recursive folder processing: scan input directories and subfolders, mirror structure in output
- Show pattern: concatenate finalized calls in order, insert tones, and document order in a text file
- CLAP annotation pattern: After normalization, run CLAP on the combined mono (out-) file using relevant prompts; filter results by confidence; add accepted annotations to the transcript and manifest
- Diarization/transcription pattern: Segment normalized vocals per speaker, store in speakers/SXX/ folders, transcribe each segment, save .txt alongside audio, and rename files with index and short transcription (≤ 488 chars)
- Manifest pattern: Track all segment lineage, including input file, timestamps, speaker, index, and transcription
- LLM workflow pattern: For each call, run all llm_tasks as defined in the workflow preset, render prompts, call the LLM API, save outputs, and update the manifest
- CLAP-driven segmentation pattern: For long audio, use CLAP to detect boundaries, segment into calls, process each as a single-file call, and track all lineage in the manifest
- Workflow/config separation pattern: workflows/ contains pipeline logic (JSON), config/ contains user/environment data; program loads both at runtime

## Component Relationships

- Ingestion → PII removal/renaming → Separation/Annotation/Normalization/Metadata/Remixing → Diarization prep → Segmentation → Transcription → Soundbite extraction → Show creation
- Each processing step updates file tracking, metadata, and outputs for next step
- Manifest and metadata ensure traceability and quality at every stage
- After remixing, mix vocals and instrumentals for trans_out and recv_out pairs into left and right channels, with 50% volume on instrumental track
- Produce new call files with chronological call index as filename and LLM-generated call title, sanitized of all punctuation from LLM call title response
- Optionally apply tones to call files (if not in show-mode)
- Show output: concatenate all valid calls (>10s, error-free) into a single WAV file, insert tones between calls, and include a text file listing call order, names, timestamps, and metadata
- All workflows are JSON files in workflows/ (routing, prompts, tasks, etc.), user-specific data in config/

# System Patterns

- Resume logic is now output-folder-centric, not input-centric.
- All job creation and file discovery for resume is based on the anonymized `renamed/` directory.
- Console output for available folders is sanitized and only shows anonymized run folder names. 