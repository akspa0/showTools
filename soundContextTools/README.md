# Audio Context Tool

A privacy-focused, modular pipeline for processing phone call audio and other recordings. Automates ingestion, PII removal, file tracking, audio separation, CLAP annotation, loudness normalization, speaker diarization, transcription, soundbite extraction, LLM integration, remixing, and show creation. All steps are orchestrated for strict privacy, traceability, and manifest/logging requirements.

---

## Features
- **Privacy-first:** No PII in filenames, logs, or outputs. All logging/manifesting is anonymized.
- **Modular pipeline:** Ingestion, separation, CLAP, diarization, normalization, transcription, soundbite, remix, show, LLM, and more.
- **CLAP-based segmentation:** Detects call boundaries in long audio using CLAP (configurable prompts, thresholds).
- **Speaker diarization:** Segments audio by speaker, with per-speaker transcripts and outputs.
- **LLM integration:** Workflow-driven LLM tasks (titles, synopses, categories, image prompts, songs, etc.)
- **Batch processing:** Handles large folders of audio, tuples, or single files (including YouTube/URL inputs).
- **Extensible:** All workflows and prompts are JSON-configurable in the `workflows/` folder.
- **Traceability:** Full manifest and metadata lineage for every file and output.

---

## Directory Structure
```
soundContextTools/
  memory-bank/           # Project docs, context, and rules
  outputs/               # All run outputs (timestamped folders)
  workflows/             # All pipeline, CLAP, and LLM configs (JSON)
  ...                    # Pipeline scripts and modules
```

---

## Installation

### 1. Using Conda (Recommended for PyTorch)
```sh
conda create -n soundcontext python=3.10
conda activate soundcontext
# Install PyTorch (choose the right CUDA version for your system):
# See https://pytorch.org/get-started/locally/ for the latest command.
# Example (CPU only):
conda install pytorch torchaudio cpuonly -c pytorch
# Example (CUDA 11.8):
conda install pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# Install the rest:
pip install -r requirements.txt
```

### 2. Using Pip Only
- Install the correct torch/torchaudio version for your system from [PyTorch.org](https://pytorch.org/get-started/locally/)
- Then:
```sh
pip install -r requirements.txt
```

---

## Configuration
- **CLAP segmentation:** `workflows/clap_segmentation.json` (prompts, thresholds, pairing logic)
- **CLAP annotation:** `workflows/clap_annotation.json` (prompts, thresholds, chunking)
- **LLM tasks:** `workflows/llm_tasks.json` (task list, prompts, model config)
- **All configs are editable JSON files.**

---

## Usage

### Basic CLI Example
```sh
python pipeline_orchestrator.py <input_dir>
```

### With CLAP-based Call Segmentation
```sh
python pipeline_orchestrator.py <input_dir> --call-cutter
```

### Processing a YouTube/URL Input
```sh
python pipeline_orchestrator.py --url "https://www.youtube.com/watch?v=..."
```

### Resume/Debug
```sh
python pipeline_orchestrator.py --output-folder outputs/run-YYYYMMDD-HHMMSS --resume
```

### Other CLI Options
- `--asr_engine parakeet|whisper` (choose ASR model)
- `--llm_config workflows/llm_tasks.json` (custom LLM task config)
- `--call-tones` (insert tones between calls in show output)
- `--resume-from <stage>` (resume from a specific stage)

---

## Customization
- **CLAP/LLM prompts, thresholds, and logic** are fully tweakable in the `workflows/` JSON files.
- Change prompts, add/remove tasks, adjust thresholds, and rerun the pipeline—no code changes needed.

---

## Troubleshooting
- **PyTorch install issues:** Use conda and follow the [official instructions](https://pytorch.org/get-started/locally/).
- **No segments detected:** Lower the CLAP confidence threshold or add more prompts in `clap_segmentation.json`.
- **Too many/false segments:** Raise the threshold or adjust pairing/gap settings.
- **LLM token limit errors:** The pipeline will chunk transcripts by speaker/segment automatically.
- **Manifest/logs:** Check the output run folder for detailed logs and manifest.json.

---

## Contributing & Support
- PRs and issues welcome!
- For questions, open an issue or contact the maintainer.

---

**Happy hacking!** 