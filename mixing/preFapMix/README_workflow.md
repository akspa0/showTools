# PreFapMix Audio Processing Workflow

This document describes the **end-to-end workflow** for processing audio files and call data using the PreFapMix pipeline. It covers single-file and batch processing, LLM-powered summarization, MP3 output, and the structure of all outputs.

---

## Overview

The PreFapMix pipeline is designed to:
- Process single or paired audio files (e.g., call center `recv_out`/`trans_out` pairs or arbitrary audio).
- Separate vocals/instrumentals, normalize, diarize, and transcribe audio.
- Generate detailed transcripts and per-speaker soundbites (now MP3 with ID3/JSON metadata).
- Use a local LLM (via LM Studio) to create:
  - A witty, PII-safe call/audio name
  - A concise synopsis
  - 3–5 plain English categories (tags)
  - Any number of custom LLM tasks (fully extensible)
- Output user-friendly, PII-safe, MP3-compressed audio and organized metadata.
- Support batch and single-file workflows.
- Always produce a clean, user-facing output folder with all results, metadata, and a zipped archive.

---

## 1. Quick Start: Single File or Batch Processing

### **A. Single File Processing**

```sh
python workflow_executor.py \
  --input_file path/to/your_audio.wav \
  --output_dir workspace/workflow_runs/ \
  --config_file default_audio_analysis_workflow.json \
  --log_level DEBUG
```
- This will process just that file and create a workflow run directory in `workspace/workflow_runs/`.

**Optional: CLAP-Based Call Segmentation**

If your audio file contains CLAP annotation data (e.g., telephone ringing/hang-up tones), you can enable automatic segmentation:

```sh
python workflow_executor.py \
  --input_file path/to/your_audio.wav \
  --output_dir workspace/workflow_runs/ \
  --config_file default_audio_analysis_workflow.json \
  --enable_clap_separation
```
- The pipeline will segment the audio into individual calls using the annotation data, and process each segment as a separate call (with its own naming, tagging, and outputs).

### **B. Batch Processing**

1. **Place all your audio files or call folders** in a directory (e.g., `workspace/test_audio/`).
2. **Run:**
```sh
python workflow_executor.py \
  --input_dir workspace/test_audio/ \
  --output_dir workspace/workflow_runs/ \
  --config_file default_audio_analysis_workflow.json \
  --log_level DEBUG
```
- Each file/folder will be processed in sequence.

---

## 2. Final Output: Unified, User-Facing Results

- The final output builder now always runs as the last step, producing:
  - Finalized call MP3s (with optional tones appended)
  - MP3 soundbites for each segment, with full ID3 and JSON metadata
  - All LLM outputs and transcripts, organized by call
  - A show file (MP3) built from the finalized calls, with a show transcript and timestamps
  - A comprehensive metadata JSON (`final_output_metadata.json`) with all paths relative to the project root
  - A zipped archive of the entire output folder for easy sharing or archiving
- The output folder is always clean, logically organized, and ready for dataset use or review.

---

## 3. Output Structure

### **A. Workflow Run Directory (per file/call)**
- Created in `workspace/workflow_runs/`.
- Contains subfolders for each processing stage (e.g., `00_clap_event_annotation/`, `03_transcription/`).
- Each stage outputs its results (e.g., separated stems, transcripts, soundbites).

### **B. Processed Calls Directory**
- Created in `workspace/processed_calls/`.
- For each call/audio:
  - Mixed or single vocal audio (`.wav`)
  - Merged or single transcript (`.json`)
  - LLM outputs (name, synopsis, tags, and any custom tasks)
  - Per-speaker soundbite folders (e.g., `RECV_S0/`, `TRANS_S1/`)

### **C. Final Output Directory (User-Facing)**
- Created in `05_final_output/` (or as configured).
- For each call/audio (folder named with sanitized LLM-generated name or fallback):
  - `[Sanitized_Name]_finalized.mp3` (primary audio, MP3-compressed, with optional tone)
  - `transcript.txt` (plain text transcript)
  - `synopsis.txt` (LLM-generated synopsis)
  - `hashtags.txt` (LLM-generated categories/tags)
  - `call_type.txt` (e.g., "pair", "single_recv")
  - `soundbites/` (MP3, with ID3/JSON metadata)
- At the root of the output folder:
  - `show_audio.mp3` (concatenated show file)
  - `show_transcript.txt` (combined transcript)
  - `show_timestamps.txt` (call start times in show)
  - `final_output_metadata.json` (all metadata, all paths relative)
  - `final_output.zip` (zipped archive of the entire output)

---

## 4. LLM-Powered Outputs: Fully Flexible

- The pipeline supports **arbitrary LLM tasks** per call or audio file, defined in the workflow JSON.
- Add as many tasks as you want (e.g., witty names, synopses, tags, image prompts, silly songs).
- Each output is saved in the LLM output directory for each call/audio.
- Edit your workflow JSON's `llm_summary_and_analysis` stage to add new tasks.

---

## 5. Command-Line Arguments (Summary)

### **workflow_executor.py**
| Argument            | Description                                      | Example                                 |
|---------------------|--------------------------------------------------|-----------------------------------------|
| `--input_file`      | Path to a single audio file to process            | `path/to/your_audio.wav`                |
| `--input_dir`       | Directory with audio files or call folders        | `workspace/test_audio/`                 |
| `--output_dir`      | Where to write workflow run outputs               | `workspace/workflow_runs/`              |
| `--config_file`     | Path to workflow JSON config                      | `default_audio_analysis_workflow.json`   |
| `--log_level`       | (Optional) Logging level                         | `DEBUG`                                 |
| `--enable_clap_separation` | Enable CLAP-based call segmentation after annotation | `True` or `False`                       |

---

## 6. Using the Final Output

- The final output folder (`05_final_output/` by default) contains everything you need:
  - All finalized call MP3s, soundbites, LLM outputs, transcripts, show file, and metadata.
  - The zipped archive (`final_output.zip`) is ready for sharing or archiving.
  - The metadata JSON (`final_output_metadata.json`) provides a complete map of all outputs, with all paths relative to the project root.
- Use the metadata JSON to:
  - Programmatically access any output (for dataset building, ML, or review)
  - Track lineage, tags, and all processing details

---

## 7. Planned Features: Content Moderation & Advanced Labeling

- **Content Moderation:**
  - Optional Whisper pass with word-level timestamps on the show file to censor or flag problematic words.
  - LLM-based evaluation of each call's transcript for sensibility or sensitive content.
  - Calls can be labeled or flagged in the output and metadata.
- **Advanced Labeling:**
  - Use the first or second LLM-generated tag as a folder label for calls.
  - More flexible folder and file naming based on LLM outputs.

---

## 8. Troubleshooting & Tips
- **LM Studio:** Ensure it is running and the model is loaded. If LLM outputs are error messages, check server status and model.
- **ffmpeg:** Must be installed and in PATH for audio mixing and MP3 conversion.
- **Soundbites:** Soundbites are now always MP3 with ID3/JSON metadata.
- **Reprocessing:** You can re-run the pipeline on any set of workflow run outputs to regenerate final outputs or after changing LLM/model settings.
- **Output Always Finalized:** The final output builder always runs, so you never have to manually trigger finalization.

---

## 9. Example End-to-End Workflow

```sh
# 1. Process all audio files in a directory
python workflow_executor.py \
  --input_dir workspace/test_audio/ \
  --output_dir workspace/workflow_runs/ \
  --config_file default_audio_analysis_workflow.json

# 2. Process a single file
python workflow_executor.py \
  --input_file path/to/your_audio.wav \
  --output_dir workspace/workflow_runs/ \
  --config_file default_audio_analysis_workflow.json
```

---

## 10. Support & Further Information
- See `memory-bank/` for detailed project context, architecture, and technical notes.
- For issues, check logs in each output directory and ensure all dependencies are installed.

---

**This README reflects the latest pipeline features and best practices.** 