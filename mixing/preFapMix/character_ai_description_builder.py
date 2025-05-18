import os
import json
import logging
from pathlib import Path
from typing import List
from llm_module import generate_llm_summary  # Assumes same interface as main script
from pydub import AudioSegment
import tempfile
import string
from pydub.utils import mediainfo

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Summarized/segmented system prompt based on character_ai_guidelines.md
CHARACTER_AI_PROMPT = """
You are an expert at writing Character.AI persona definitions. Given a transcript of a single speaker, generate a Character.AI-compatible character description. Focus on:
- Name (use SXX or a placeholder if not available)
- Short and long description
- Greeting
- Personality traits and quirks
- Example dialog (at least 2 exchanges)
- JSON formatting if possible
Be concise and use only information from the transcript. Do not invent details not present in the transcript.
"""

# Helper to aggregate all .txt files in a speaker folder
def aggregate_speaker_transcript(speaker_dir: Path) -> str:
    texts = []
    for txt_file in sorted(speaker_dir.glob('*.txt')):
        with open(txt_file, 'r', encoding='utf-8') as f:
            texts.append(f.read().strip())
    return '\n'.join(texts)

# Helper to create audio clips of given durations (in seconds)
def create_audio_clips(speaker_dir: Path, out_dir: Path, speaker_label: str):
    soundbites = sorted(speaker_dir.glob('*.wav'))
    if not soundbites:
        logging.warning(f"No soundbites for {speaker_label} in {speaker_dir}")
        return
    durations = [15, 30, 60]
    for target_sec in durations:
        combined = AudioSegment.empty()
        total_ms = 0
        for wav_file in soundbites:
            if not is_valid_audio(wav_file):
                continue
            seg = AudioSegment.from_wav(wav_file)
            if total_ms + len(seg) > target_sec * 1000:
                seg = seg[:max(0, target_sec * 1000 - total_ms)]
            combined += seg
            total_ms += len(seg)
            if total_ms >= target_sec * 1000:
                break
        if total_ms == 0:
            logging.warning(f"Not enough audio for {speaker_label} to create {target_sec}s clip.")
            continue
        out_mp3 = out_dir / f"{speaker_label}_{target_sec}s.mp3"
        combined.export(out_mp3, format="mp3")
        logging.info(f"Wrote {target_sec}s audio clip for {speaker_label} at {out_mp3}")
        if total_ms < target_sec * 1000:
            logging.warning(f"{speaker_label}: Only {total_ms/1000:.1f}s available for {target_sec}s clip.")

def is_valid_audio(file_path, min_duration_sec=5):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        logging.error(f"File does not exist or is empty: {file_path}")
        return False
    try:
        info = mediainfo(file_path)
        duration = float(info.get('duration', 0))
        if duration < min_duration_sec:
            logging.error(f"File too short (<{min_duration_sec}s): {file_path} (duration: {duration:.2f}s)")
            return False
    except Exception as e:
        logging.error(f"Could not get audio info for {file_path}: {e}")
        return False
    return True

# Main processing function
def build_character_ai_descriptions(input_root: Path, output_root: Path, llm_model: str, llm_endpoint: str):
    logging.info(f"LLM endpoint: {llm_endpoint}")
    logging.info(f"LLM model: {llm_model}")
    call_folders = [p for p in input_root.rglob('03_transcription')]
    for transcription_dir in call_folders:
        call_folder = transcription_dir.parent
        call_name = call_folder.name
        logging.info(f"Processing call: {call_name}")
        for speaker_dir in [d for d in transcription_dir.iterdir() if d.is_dir() and d.name.startswith('S')]:
            speaker_label = speaker_dir.name
            logging.info(f"  Processing speaker: {speaker_label}")
            speaker_out_dir = output_root / call_name / 'character_ai_descriptions' / speaker_label
            speaker_out_dir.mkdir(parents=True, exist_ok=True)
            transcript = aggregate_speaker_transcript(speaker_dir)
            if not transcript.strip():
                logging.warning(f"No transcript for {speaker_label} in {call_name}")
                continue
            prompt = CHARACTER_AI_PROMPT + f"\n\nTranscript for {speaker_label}:\n" + transcript
            logging.info(f"    LLM prompt (truncated): {prompt[:300]}{'...' if len(prompt) > 300 else ''}")
            try:
                # Write transcript to a temp file
                with tempfile.NamedTemporaryFile('w+', delete=False, suffix='.txt', encoding='utf-8') as tf:
                    tf.write(transcript)
                    transcript_path = Path(tf.name)
                llm_config = {
                    'lm_studio_model_identifier': llm_model,
                    'lm_studio_base_url': llm_endpoint,
                    'max_response_tokens': 2048,
                    'temperature': 0.7
                }
                output_filename = f"{speaker_label}.txt"
                output_dir = speaker_out_dir
                output_path = generate_llm_summary(
                    transcript_path,
                    CHARACTER_AI_PROMPT,
                    llm_config,
                    output_dir,
                    output_filename
                )
                # Read the response for logging
                if output_path and output_path.exists():
                    with open(output_path, 'r', encoding='utf-8') as f:
                        llm_response = f.read()
                    if not llm_response or not llm_response.strip():
                        logging.error(f"    LLM response is empty for {speaker_label} in {call_name}")
                    else:
                        logging.info(f"    LLM response received (truncated): {llm_response[:200]}{'...' if len(llm_response) > 200 else ''}")
                else:
                    logging.error(f"    LLM summary file was not created for {speaker_label} in {call_name}")
            except Exception as e:
                logging.error(f"    LLM failed for {speaker_label} in {call_name}: {e}")
            # Audio clips
            try:
                create_audio_clips(speaker_dir, speaker_out_dir, speaker_label)
            except Exception as e:
                logging.error(f"    Audio clip creation failed for {speaker_label} in {call_name}: {e}")

def cleanup_small_folders(output_root: Path, min_size_kb: int = 192):
    for call_dir in output_root.iterdir():
        if call_dir.is_dir():
            total_size = sum(f.stat().st_size for f in call_dir.rglob('*') if f.is_file())
            if total_size < min_size_kb * 1024:
                import shutil
                shutil.rmtree(call_dir)
                logging.info(f"Removed small output folder: {call_dir} (size: {total_size/1024:.1f} KB)")

def sanitize_call_name(name: str) -> str:
    # Remove punctuation, limit to 8 words, replace spaces with underscores
    name = name.translate(str.maketrans('', '', string.punctuation))
    words = name.strip().split()
    sanitized = '_'.join(words[:8])
    return sanitized or None

def rename_folders_with_call_names(output_root: Path):
    existing_names = set()
    for call_dir in list(output_root.iterdir()):
        if not call_dir.is_dir():
            continue
        call_title_file = call_dir / 'call_title.txt'
        if call_title_file.exists():
            with open(call_title_file, 'r', encoding='utf-8') as f:
                call_name = f.read().strip()
            sanitized = sanitize_call_name(call_name) if call_name else None
        else:
            sanitized = None
        if not sanitized:
            sanitized = call_dir.name
        # Ensure uniqueness
        base_name = sanitized
        suffix = 1
        while sanitized in existing_names or (output_root / sanitized).exists():
            sanitized = f"{base_name}_{suffix}"
            suffix += 1
        existing_names.add(sanitized)
        if sanitized != call_dir.name:
            new_path = output_root / sanitized
            call_dir.rename(new_path)
            logging.info(f"Renamed output folder: {call_dir.name} → {sanitized}")

if __name__ == '__main__':
    import sys
    DEFAULT_MODEL = 'llama-3.1-8b-supernova-etherealhermes'
    DEFAULT_ENDPOINT = os.environ.get('LLM_ENDPOINT', 'http://192.168.1.131:1234/v1')
    if len(sys.argv) < 3:
        print("Usage: python character_ai_description_builder.py <input_root> <output_root> [<llm_model>] [<llm_endpoint>]")
        exit(1)
    input_root = Path(sys.argv[1]).resolve()
    output_root = Path(sys.argv[2]).resolve()
    llm_model = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_MODEL
    llm_endpoint = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_ENDPOINT
    print(f"Using LLM model: {llm_model}")
    print(f"Using LLM endpoint: {llm_endpoint}")
    build_character_ai_descriptions(input_root, output_root, llm_model, llm_endpoint)
    cleanup_small_folders(output_root)
    rename_folders_with_call_names(output_root) 