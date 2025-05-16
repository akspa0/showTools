# PreFapMix Audio Processing Workflow

This document describes the **end-to-end workflow** for processing audio files and call data using the PreFapMix pipeline. It covers single-file and batch processing, LLM-powered summarization, MP3 output, and the structure of all outputs.

---

## Overview

The PreFapMix pipeline is designed to:
- Process single or paired audio files (e.g., call center `recv_out`/`trans_out` pairs or arbitrary audio).
- Separate vocals/instrumentals, normalize, diarize, and transcribe audio.
- Generate detailed transcripts and per-speaker soundbites.
- Use a local LLM (via LM Studio) to create:
  - A witty, PII-safe call/audio name
  - A concise synopsis
  - 3–5 hashtag categories
- Output user-friendly, PII-safe, MP3-compressed audio and organized metadata.
- Support batch and single-file workflows.

---

## 1. Quick Start: Single File or Batch Processing

### **A. Single File Processing**

1. **Run:**
```sh
python workflow_executor.py \
  --input_file workspace/test_audio_single/your_audio.wav \
  --output_dir workspace/workflow_runs/ \
  --workflow_config default_audio_analysis_workflow.json \
  --log_level DEBUG
```
- This will process just that file and create a workflow run directory in `workspace/workflow_runs/`.

### **B. Batch Processing**

1. **Place all your audio files or call folders** in a directory (e.g., `workspace/test_audio/`).
2. **Run:**
```sh
python workflow_executor.py \
  --input_dir workspace/test_audio/ \
  --output_dir workspace/workflow_runs/ \
  --workflow_config default_audio_analysis_workflow.json \
  --log_level DEBUG
```
- Each file/folder will be processed in sequence.

---

## 2. Call Aggregation & Finalization (`call_processor.py`)

- **Automatic:** When `workflow_executor.py` detects a call folder (with both `recv_out` and `trans_out`), it will automatically invoke `call_processor.py`.
- **Manual (for reprocessing or debugging):**
```sh
python call_processor.py \
  --input_run_dir workspace/workflow_runs/ \
  --output_call_dir workspace/processed_calls/ \
  --final_output_dir workspace/final_processed_calls/ \
  --llm_model_id "NousResearch/Hermes-2-Pro-Llama-3-8B" \
  --log_level DEBUG
```
- `--llm_model_id` is optional; if omitted, a default model is used.
- `--final_output_dir` creates a user-friendly, flat output structure for each call/audio.

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
  - LLM outputs:
    - `[call_id]_suggested_name.txt`
    - `[call_id]_combined_call_summary.txt`
    - `[call_id]_hashtags.txt`
  - Per-speaker soundbite folders (e.g., `RECV_S0/`, `TRANS_S1/`)

### **C. Final Output Directory (User-Facing)**
- Created in `workspace/final_processed_calls/`.
- For each call/audio (folder named with sanitized LLM-generated name or fallback):
  - `[Sanitized_Name].mp3` (primary audio, MP3-compressed)
  - `transcript.txt` (plain text transcript)
  - `synopsis.txt` (LLM-generated synopsis)
  - `hashtags.txt` (LLM-generated hashtags)
  - `call_type.txt` (e.g., "pair", "single_recv")
  - (Optional) `soundbites/` (if enabled)

---

## 4. LLM Integration (LM Studio)
- **Requirements:**
  - LM Studio must be running and accessible (default: `http://localhost:1234/v1`).
  - The desired model (e.g., `NousResearch/Hermes-2-Pro-Llama-3-8B`) must be loaded in LM Studio.
- **LLM Outputs:**
  - Always generates three `.txt` files per call/audio: name, synopsis, hashtags.
  - If the LLM server is unreachable or misconfigured, files will contain error messages.

---

## 5. Command-Line Arguments (Summary)

### **workflow_executor.py**
| Argument            | Description                                      | Example                                 |
|---------------------|--------------------------------------------------|-----------------------------------------|
| `--input_file`      | Path to a single audio file to process            | `workspace/test_audio_single/your_audio.wav` |
| `--input_dir`       | Directory with audio files or call folders        | `workspace/test_audio/`                 |
| `--output_dir`      | Where to write workflow run outputs               | `workspace/workflow_runs/`              |
| `--workflow_config` | Path to workflow JSON config                      | `default_audio_analysis_workflow.json`   |
| `--log_level`       | (Optional) Logging level                         | `DEBUG`                                 |

### **call_processor.py**
| Argument             | Description                                      | Example                                 |
|----------------------|--------------------------------------------------|-----------------------------------------|
| `--input_run_dir`    | Directory with workflow run outputs               | `workspace/workflow_runs/`              |
| `--output_call_dir`  | Where to write processed call outputs             | `workspace/processed_calls/`            |
| `--final_output_dir` | (Optional) User-facing, flat output structure     | `workspace/final_processed_calls/`      |
| `--llm_model_id`     | (Optional) LLM model identifier                   | `NousResearch/Hermes-2-Pro-Llama-3-8B`  |
| `--log_level`        | (Optional) Logging level                         | `DEBUG`                                 |

---

## 6. Troubleshooting & Tips
- **LM Studio:** Ensure it is running and the model is loaded. If LLM outputs are error messages, check server status and model.
- **ffmpeg:** Must be installed and in PATH for audio mixing and MP3 conversion.
- **Soundbites:** By default, soundbites are copied as `.wav`. MP3 compression for soundbites is under consideration.
- **Reprocessing:** You can re-run `call_processor.py` on any set of workflow run outputs to regenerate final outputs or after changing LLM/model settings.

---

## 7. Advanced: Show Compilation (Planned)
- A future script (`show_compiler.py`) will allow you to concatenate multiple processed calls into a single "show" MP3, with timestamps and a combined transcript.

---

## 8. Example End-to-End Workflow

```sh
# 1. Process all audio files in a directory
python workflow_executor.py \
  --input_dir workspace/test_audio/ \
  --output_dir workspace/workflow_runs/ \
  --workflow_config default_audio_analysis_workflow.json

# 2. (Optional) Reprocess or finalize all calls
python call_processor.py \
  --input_run_dir workspace/workflow_runs/ \
  --output_call_dir workspace/processed_calls/ \
  --final_output_dir workspace/final_processed_calls/ \
  --llm_model_id "NousResearch/Hermes-2-Pro-Llama-3-8B"
```

---

## 9. Support & Further Information
- See `memory-bank/` for detailed project context, architecture, and technical notes.
- For issues, check logs in each output directory and ensure all dependencies are installed.

---

**This README reflects the latest pipeline features and best practices.** 