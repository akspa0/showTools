# PreFapMix Audio Processing Workflow

## What's New (May 2025)
- **ASR Engine Selection:** Choose between Whisper and NVIDIA Parakeet via CLI/config.
- **yt-dlp Audio URL Support:** Process audio directly from URLs using yt-dlp.
- **Output Folder Cleanup:** Any output folder under 192KB is automatically deleted after processing.
- **Robust Naming Conventions:** All stems and output folders use sanitized, PII-safe, and unique names.
- **Extensible LLM Task System:** Add any number of custom LLM tasks in the workflow config.
- **Soundbites:** Now always MP3 with ID3/JSON metadata for all speakers.
- **Error Handling & Logging:** All scripts have robust error handling and detailed logs.
- **New Scripts:** `final_output_builder.py` and `character_ai_description_builder.py` for advanced output and character description generation.

---

## Workflow Overview

## CLAP Segmentation (2025 Refactor)
- Direct use of Hugging Face `transformers` CLAP model and processor for segmentation.
- All audio I/O (duration, segment extraction) is handled via `ffmpeg`/`ffprobe` subprocess calls.
- No use of `soundfile`, `librosa`, or similar Python audio libraries for segmentation.
- All legacy `CLAPAnnotatorWrapper`, CLI, and stem separation logic has been removed from segmentation.
- This is now the canonical approach for segmentation in the pipeline.

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

## 1. Quick Start: Single File, Batch, or URL Processing

### **A. Single File Processing (with ASR selection)**
```sh
python workflow_executor.py \
  --input_file path/to/your_audio.wav \
  --output_dir workspace/workflow_runs/ \
  --config_file default_audio_analysis_workflow.json \
  --asr_engine whisper   # or --asr_engine parakeet
```

### **B. Batch Processing**
```sh
python workflow_executor.py \
  --input_dir workspace/test_audio/ \
  --output_dir workspace/workflow_runs/ \
  --config_file default_audio_analysis_workflow.json \
  --asr_engine parakeet   # or --asr_engine whisper
```

### **C. Process Audio from a URL (yt-dlp)**
```sh
python workflow_executor.py \
  --url "https://www.youtube.com/watch?v=..." \
  --output_dir workspace/workflow_runs/ \
  --config_file default_audio_analysis_workflow.json \
  --asr_engine whisper   # or --asr_engine parakeet
```

---

### **ASR Engine Selection via CLI**
- Use `--asr_engine whisper` or `--asr_engine parakeet` to select the transcription engine for your run.
- This flag **overrides** the `asr_engine` value in the workflow config for all files in the run.
- Works for both single-file and batch processing.
- If not specified, the engine in the workflow config is used.

---

## 2. Final Output Builder (Manual Invocation)
The final output builder is usually run automatically, but can be invoked manually:
```sh
python final_output_builder.py \
  --input_dir workspace/processed_calls/ \
  --output_dir 05_final_output/
```
- Cleans, renames, and organizes all outputs for user-facing delivery.
- Deletes any output folder under 192KB.
- Produces zipped archive and metadata JSON.

---

## 3. Character.AI Description Builder
Generate Character.AI-style descriptions and audio clips for each speaker:
```sh
python character_ai_description_builder.py \
  --input_dir 05_final_output/ \
  --llm_endpoint http://localhost:1234/v1 \
  --llm_model llama-3.1-8b-supernova-etherealhermes
```
- Creates 15s, 30s, 60s MP3 clips per speaker with ID3 tags.
- Generates LLM-based character descriptions.
- Cleans up small/empty output folders.

---

## 4. Output Structure & Naming Conventions
- All folders/files are sanitized for PII and filesystem safety.
- Stems: `<callname>_RECV_vocals.wav`, `<callname>_TRANS_instrumental.wav`, etc.
- Output folders: Named using sanitized LLM-generated call names (fallback to call_id, uniqueness enforced).
- Any output folder under 192KB is deleted.
- Soundbites: MP3, ID3/JSON tagged, organized by speaker.

---

## 5. Extensible LLM Task System (Example)
Add custom LLM tasks in your workflow config:
```json
{
  "stage_name": "llm_summary_and_analysis",
  "module": "llm_module",
  "function": "run_llm_tasks",
  "inputs": { ... },
  "config": {
    "llm_tasks": [
      {"name": "call_title", "prompt_template": "...", "output_file": "call_title.txt"},
      {"name": "image_prompt", "prompt_template": "...", "output_file": "image_prompt.txt"},
      {"name": "silly_song", "prompt_template": "...", "output_file": "silly_song.txt"}
    ],
    ...
  }
}
```
- Outputs are saved in the LLM output directory for each call/audio.
- Add/remove tasks by editing the workflow JSON.

---

## 6. Troubleshooting & Tips
- **ASR Engine:** Use `--asr_engine whisper` or `--asr_engine parakeet` as needed (overrides config file for the run).
- **yt-dlp:** Ensure yt-dlp is installed for URL support.
- **Output Cleanup:** Folders under 192KB are deleted—check logs if outputs are missing.
- **Naming:** All names are sanitized; check logs for fallback naming.
- **Logs:** Detailed logs are written in each output directory.
- **Dependencies:** ffmpeg, yt-dlp, and all Python requirements must be installed.

---

## 7. Example End-to-End Workflow
```sh
# Batch process audio files
ython workflow_executor.py \
  --input_dir workspace/test_audio/ \
  --output_dir workspace/workflow_runs/ \
  --config_file default_audio_analysis_workflow.json

# Build final output (if needed)
python final_output_builder.py \
  --input_dir workspace/processed_calls/ \
  --output_dir 05_final_output/

# Generate Character.AI descriptions
python character_ai_description_builder.py \
  --input_dir 05_final_output/ \
  --llm_endpoint http://localhost:1234/v1 \
  --llm_model llama-3.1-8b-supernova-etherealhermes
```

---

## 8. Support & Further Information
- See `memory-bank/` for detailed project context, architecture, and technical notes.
- For issues, check logs in each output directory and ensure all dependencies are installed.

---

**This README reflects the latest pipeline features, usage, and best practices.** 