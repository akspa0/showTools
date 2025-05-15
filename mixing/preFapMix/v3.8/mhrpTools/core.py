import os
import subprocess
import logging
from pathlib import Path
import json
import re
import hashlib
from pydub import AudioSegment
import yaml
import datetime
import lmstudio as lms
import soundfile as sf
import shutil

logging.basicConfig(level=logging.INFO)

# Default CLAP prompts and separator model (can be customized)
DEFAULT_CLAP_PROMPTS = "telephone noises, dial tone, ring tone, telephone interference"
DEFAULT_SEPARATOR_MODEL = "Mel Band RoFormer Vocals"
DEFAULT_CONFIDENCE = 0.6
MAX_NAME_LEN = 32
TEMP_PROCESSING_DIR = Path("./temp_processing")
TEMP_PROCESSING_DIR.mkdir(exist_ok=True)

# Helper: Aggressively shorten output names
def short_job_id(filename):
    # Extract last 4, then 2 numeric tokens
    numbers = re.findall(r'\d+', filename)
    if len(numbers) >= 2:
        shortid = '_'.join(numbers[-2:])
    elif numbers:
        shortid = numbers[-1]
    else:
        shortid = filename[:8]
    # If still too long, hash the filename
    if len(shortid) > MAX_NAME_LEN:
        shortid = hashlib.sha1(filename.encode()).hexdigest()[:8]
    return shortid[:MAX_NAME_LEN]

# Helper: Validate audio file (non-empty, can be opened)
def is_valid_audio_file(filepath):
    # Skip files under 40KB (likely just header or empty)
    try:
        if not os.path.isfile(filepath):
            return False
        if os.path.getsize(filepath) < 40 * 1024:
            return False
        return True
    except Exception:
        return False

# Run ClapAnnotator via script (not as a module)
def run_clap_annotator(input_path, call_folder, prompts=DEFAULT_CLAP_PROMPTS, separator_model=DEFAULT_SEPARATOR_MODEL, confidence=DEFAULT_CONFIDENCE, **kwargs):
    """
    Call ClapAnnotator CLI to process input audio in a local temp directory, then move/copy outputs to call_folder.
    Returns the call_folder where results are saved.
    """
    temp_dir = TEMP_PROCESSING_DIR / f"clap_{os.getpid()}"
    temp_dir.mkdir(exist_ok=True)
    try:
        logging.info(f"[mhrpTools] Running ClapAnnotator on {input_path} (temp output: {temp_dir})")
        cmd = [
            "python", "ClapAnnotator/cli.py",
            str(input_path),
            "--prompts", prompts,
            "--separator-model", separator_model,
            "--confidence", str(confidence),
            "--output-dir", str(temp_dir),
            "--keep-audio"
        ]
        subprocess.run(cmd, check=True)
        # Move/copy outputs to call_folder with short names
        for f in temp_dir.glob("*_(Vocals)*.wav"):
            shutil.copy2(f, call_folder / "vocals.wav")
        for f in temp_dir.glob("*_(Instrumental)*.wav"):
            shutil.copy2(f, call_folder / "instrumental.wav")
        for f in temp_dir.glob("*clap_results.json"):
            shutil.copy2(f, call_folder / "clap_results.json")
        logging.info(f"[mhrpTools] Consolidated ClapAnnotator outputs to {call_folder}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return call_folder

# Run preFapMix mixing via script
def run_prefapmix_mixing(input_dir, output_dir, **kwargs):
    """
    Call preFapMix.py to process recv_out/trans_out files for stereo mixing.
    """
    logging.info(f"[mhrpTools] Running preFapMix mixing on {input_dir}")
    cmd = [
        "python", "preFapMix/preFapMix.py",
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir)
    ]
    subprocess.run(cmd, check=True)
    return output_dir

# Run WhisperBite via script
def run_whisperbite(input_path, output_dir, model="large-v3", num_speakers=2, **kwargs):
    """
    Call WhisperBite CLI to process audio for transcription/soundbites.
    Demucs/vocal separation is disabled.
    """
    logging.info(f"[mhrpTools] Running WhisperBite on {input_path}")
    cmd = [
        "python", "WhisperBite/whisperBite.py",
        "--input_file", str(input_path),
        "--output_dir", str(output_dir),
        "--model", model,
        "--num_speakers", str(num_speakers)
        # Do NOT add --enable_vocal_separation
    ]
    subprocess.run(cmd, check=True)
    return output_dir

# Merge CLAP annotations into master transcript
def merge_clap_annotations_with_transcript(clap_output_dir, whisperbite_output_dir):
    """
    Find CLAP results.json and WhisperBite master_transcript.txt, and merge them.
    Appends CLAP annotations as a section at the end of the transcript.
    """
    # Find results.json (CLAP)
    results_json = None
    for f in Path(clap_output_dir).rglob("results.json"):
        results_json = f
        break
    if not results_json:
        logging.warning("No CLAP results.json found for annotation merging.")
        return
    # Find master_transcript.txt (WhisperBite)
    master_transcript = None
    for f in Path(whisperbite_output_dir).rglob("master_transcript.txt"):
        master_transcript = f
        break
    if not master_transcript:
        logging.warning("No WhisperBite master_transcript.txt found for annotation merging.")
        return
    # Load both
    with open(results_json, "r", encoding="utf-8") as f:
        clap_data = json.load(f)
    with open(master_transcript, "a", encoding="utf-8") as f:
        f.write("\n\n=== CLAP Annotations ===\n")
        json.dump(clap_data, f, indent=2)
    logging.info(f"Merged CLAP annotations into {master_transcript}")

def get_relative_output_dir(input_file, input_root, output_root):
    """
    Given an input file, input root, and output root, return the output directory path that mirrors the input structure,
    with a job_<shortid> leaf.
    """
    input_file = Path(input_file).resolve()
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()
    try:
        rel_path = input_file.relative_to(input_root)
    except ValueError:
        # If not under input_root, just use the filename
        rel_path = input_file.name
    # Remove the filename, keep the parent dirs
    if isinstance(rel_path, Path):
        rel_dir = rel_path.parent
    else:
        rel_dir = Path("")
    job_id = short_job_id(input_file.stem)
    out_dir = output_root / rel_dir / f"job_{job_id}"
    return out_dir

def strip_prefix(filename):
    # Remove recv_out, trans_out, or out prefix from filename
    return re.sub(r'^(recv_out|trans_out|out)[-_]?', '', filename)

def ffmpeg_normalize(input_file, output_file):
    """Normalize audio to -14 LUFS using FFmpeg loudnorm."""
    cmd = [
        'ffmpeg', '-y', '-i', str(input_file),
        '-af', 'loudnorm=I=-14:TP=-1.5:LRA=11',
        str(output_file)
    ]
    logging.info(f"[mhrpTools] Normalizing {input_file} -> {output_file}")
    subprocess.run(cmd, check=True)

def ffmpeg_stereo_mix(incoming_file, outgoing_file, output_file):
    """Mix incoming and outgoing mono files into a stereo file with 40% separation using FFmpeg pan filter."""
    cmd = [
        'ffmpeg', '-y',
        '-i', str(incoming_file),
        '-i', str(outgoing_file),
        '-filter_complex',
        '[0:a]loudnorm=I=-14:TP=-1.5:LRA=11[a0];'
        '[1:a]loudnorm=I=-14:TP=-1.5:LRA=11[a1];'
        '[a0][a1]amerge=inputs=2,pan=stereo|c0=0.6*c0+0.4*c1|c1=0.6*c1+0.4*c0[out]',
        '-map', '[out]',
        '-ac', '2',
        str(output_file)
    ]
    logging.info(f"[mhrpTools] Stereo mixing (40% separation): {incoming_file} + {outgoing_file} -> {output_file}")
    subprocess.run(cmd, check=True)

def ffmpeg_concat(files, output_file):
    """Concatenate a list of stereo files into a single file using FFmpeg concat demuxer."""
    list_file = output_file.parent / 'concat_list.txt'
    with open(list_file, 'w') as f:
        for file in files:
            f.write(f"file '{file}'\n")
    cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', str(list_file),
        '-c', 'copy',
        str(output_file)
    ]
    logging.info(f"[mhrpTools] Concatenating {len(files)} files -> {output_file}")
    subprocess.run(cmd, check=True)

def get_root_output_folder(input_folder):
    """Return output folder as <input_folder_name>_<timestamp>"""
    input_folder = Path(input_folder)
    show_name = input_folder.name
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    return Path(f"{show_name}_{timestamp}")

def llm_generate_call_title(transcript, model_name="nidum-gemma-3-4b-it-uncensored"):
    """Use a local LLM to generate a short, descriptive title for a call based on its transcript."""
    prompt = (
        "Here is a transcript of a phone call. Please generate a short, descriptive, and catchy title for this call. "
        "The title should be 3-8 words, summarize the main theme or most memorable moment, and be suitable for use as a filename. "
        "Avoid generic titles like 'Phone Call' or 'Conversation'.\n\nTranscript:\n" + transcript
    )
    try:
        model = lms.llm(model_name)
        result = model.respond(prompt)
        # Sanitize for filesystem safety
        title = result.strip()
        title = re.sub(r'[^\w\- ]', '', title)  # Remove unsafe chars
        title = re.sub(r'\s+', '_', title)[:40]  # Replace spaces with underscores, limit length
        return title or None
    except Exception as e:
        logging.error(f"[mhrpTools] LLM call naming failed: {e}")
        return None

def validate_and_repair_audio(input_file):
    """Validate audio file for channel length/sample rate. Attempt auto-repair with FFmpeg if needed."""
    try:
        # Check with soundfile for sample rate and channels
        info = sf.info(str(input_file))
        if info.frames == 0 or info.samplerate < 8000:
            logging.warning(f"[mhrpTools] {input_file} is empty or has invalid sample rate.")
            return False, None
        # Check with pydub for channel length consistency
        audio = AudioSegment.from_file(str(input_file))
        if audio.channels == 2:
            left = audio.split_to_mono()[0]
            right = audio.split_to_mono()[1]
            if len(left) != len(right):
                logging.warning(f"[mhrpTools] {input_file} stereo channels have mismatched lengths.")
                # Attempt auto-repair
                fixed_file = Path(str(input_file).replace('.wav', '_fixed.wav'))
                cmd = [
                    'ffmpeg', '-y', '-i', str(input_file),
                    '-ac', '2', '-ar', '44100', str(fixed_file)
                ]
                subprocess.run(cmd, check=True)
                # Re-validate
                audio_fixed = AudioSegment.from_file(str(fixed_file))
                if len(audio_fixed.split_to_mono()[0]) == len(audio_fixed.split_to_mono()[1]):
                    logging.info(f"[mhrpTools] Auto-repaired {input_file} to {fixed_file}.")
                    return True, fixed_file
                else:
                    logging.error(f"[mhrpTools] Auto-repair failed for {input_file}.")
                    return False, None
        # If mono or stereo and valid
        return True, None
    except Exception as e:
        logging.error(f"[mhrpTools] Audio validation failed for {input_file}: {e}")
        return False, None

def extract_timestamp_from_filename(filename):
    import re
    # Look for a YYYYMMDD or similar pattern in the filename
    match = re.search(r'(20\d{6})[_-]?(\d{4})?', filename)
    if match:
        date = match.group(1)
        time = match.group(2) if match.group(2) else '0000'
        return f"{date}_{time}"
    # Fallback: use current datetime
    import datetime
    return datetime.datetime.now().strftime('%Y%m%d_%H%M')

def llm_generate_image_prompt(transcript, model_name="nidum-gemma-3-4b-it-uncensored"):
    system_prompt = (
        "You are an AI assistant specialized in generating detailed and creative image prompts for AI image generation. "
        "Your task is to expand a given user prompt into a well-structured, vivid, and highly descriptive prompt, while limiting the overall response to no more than 120 words total, without truncating the prompt. "
        "Enhance the visual quality and artistic impact by adding relevant details but ensure everyone is an adult. Do not omit or alter any key elements provided. "
        "Follow the given instructions or guidelines and respond only with the refined prompt."
    )
    prompt = f"{system_prompt}\n\nUser prompt:\n{transcript}"
    try:
        model = lms.llm(model_name)
        result = model.respond(prompt)
        return result.strip()
    except Exception as e:
        logging.error(f"[mhrpTools] LLM image prompt generation failed: {e}")
        return None

# Main pipeline function
def mhrp_pipeline(input_path, output_dir=None, mode='auto', process_left_channel=False, skipped_files=None, parse_out_files=False, workflow_preset='show-edit', **kwargs):
    """
    Orchestrate the full pipeline:
    1. Run ClapAnnotator on input audio (unless filtered).
    2. If output files are recv_out/trans_out, run preFapMix mixing.
    3. For soundbites, only process trans_out-prefixed files with (Vocals) in the filename by default.
    4. If process_left_channel is True, also process recv_out-prefixed files with (Vocals) in the filename.
    5. For mixing, only process the original normalized trans_out and recv_out files.
    6. Merge CLAP annotations into the final master transcript after WhisperBite.
    7. Aggressively shorten output folder/file names to avoid PII and path length issues.
    8. Skip and log invalid/empty audio files.
    9. By default, skip files prefixed with 'out' unless parse_out_files is True.
    10. All outputs for a given input file go into a single output folder, with soundbites in a 'wb' subfolder, left/right labeled.
    workflow_preset: controls the pipeline. 'show-edit' disables tones at the end of calls, inserts tones.wav only between calls in the show.
    """
    if skipped_files is None:
        skipped_files = []
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine input_root (for batch: the original input_path, for single: input_path.parent)
    if input_path.is_dir():
        input_root = input_path
    else:
        input_root = input_path.parent

    # In batch processing, filter out 'out'-prefixed files unless parse_out_files is True
    if input_path.is_dir():
        audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        for file in input_path.rglob('*'):
            if file.is_file() and file.suffix.lower() in audio_exts:
                # Skip files inside the output directory to prevent recursion
                try:
                    file.relative_to(output_dir)
                    continue  # This file is inside the output directory
                except ValueError:
                    pass  # Not inside output_dir, process it
                # Filter out 'out'-prefixed files unless parse_out_files is True
                if file.name.startswith('out') and not parse_out_files:
                    logging.info(f"[mhrpTools] Skipping {file} (filtered 'out'-prefix, parse_out_files=False)")
                    skipped_files.append(f"{file} (filtered 'out'-prefix)")
                    continue
                # Validate audio file
                if not is_valid_audio_file(file):
                    logging.warning(f"Skipping invalid or empty audio file: {file}")
                    skipped_files.append(str(file))
                    continue
                # Output folder: output_root/strip_prefix(input_basename)
                base_name = strip_prefix(file.stem)
                job_output_dir = output_dir / base_name
                job_output_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f"[mhrpTools] Mapping input {file} to output {job_output_dir}")
                mhrp_pipeline(file, job_output_dir, mode=mode, process_left_channel=process_left_channel, skipped_files=skipped_files, parse_out_files=parse_out_files)
        if skipped_files:
            logging.warning(f"Skipped files: {skipped_files}")
        return skipped_files

    # For single file: filter out 'out'-prefix unless parse_out_files is True
    if input_path.name.startswith('out') and not parse_out_files:
        logging.info(f"[mhrpTools] Skipping {input_path} (filtered 'out'-prefix, parse_out_files=False)")
        if skipped_files is not None:
            skipped_files.append(f"{input_path} (filtered 'out'-prefix)")
        return
    # Validate single file
    if not is_valid_audio_file(input_path):
        msg = f"[mhrpTools] Skipping {input_path}: file is under 40KB or not a valid audio file."
        print(msg)
        if skipped_files is not None:
            skipped_files.append(f"{input_path} (too small/invalid)")
        return
    # Output folder: output_root/strip_prefix(input_basename)
    base_name = strip_prefix(input_path.stem)
    job_output_dir = output_dir / base_name
    job_output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"[mhrpTools] Mapping input {input_path} to output {job_output_dir}")
    # Step 1: Validate and repair audio
    valid, fixed_file = validate_and_repair_audio(input_path)
    if not valid:
        logging.error(f"[mhrpTools] Skipping {input_path}: failed audio validation and repair.")
        if skipped_files is not None:
            skipped_files.append(f"{input_path} (failed validation/repair)")
        return
    if fixed_file:
        input_path = fixed_file
    # Step 2: Run ClapAnnotator
    call_folder = run_clap_annotator(input_path, job_output_dir)
    # Step 3: Check for recv_out/trans_out files in the output
    files = list(job_output_dir.rglob("*.wav"))
    has_recv = any(f.name.startswith("recv_out") for f in files)
    has_trans = any(f.name.startswith("trans_out") for f in files)
    # --- Mixing: Only process original normalized trans_out/recv_out files ---
    if mode in ("mixing", "both") or (mode == "auto" and has_recv and has_trans):
        prefapmix_output_dir = job_output_dir / "mix"
        prefapmix_output_dir.mkdir(exist_ok=True)
        run_prefapmix_mixing(job_output_dir, prefapmix_output_dir)
        right_files = list((prefapmix_output_dir / "right").rglob("*.wav"))
        for right_file in right_files:
            if right_file.name.startswith("trans_out") and "(Vocals)" in right_file.name:
                wb_base = strip_prefix(right_file.stem)
                wb_dir = job_output_dir / "wb" / wb_base / "right"
                wb_dir.mkdir(parents=True, exist_ok=True)
                run_whisperbite(right_file, wb_dir)
                merge_clap_annotations_with_transcript(job_output_dir, wb_dir)
        if process_left_channel:
            left_files = list((prefapmix_output_dir / "left").rglob("*.wav"))
            for left_file in left_files:
                if left_file.name.startswith("recv_out") and "(Vocals)" in left_file.name:
                    wb_base = strip_prefix(left_file.stem)
                    wb_dir = job_output_dir / "wb" / wb_base / "left"
                    wb_dir.mkdir(parents=True, exist_ok=True)
                    run_whisperbite(left_file, wb_dir)
                    merge_clap_annotations_with_transcript(job_output_dir, wb_dir)
        if mode == "both":
            stereo_files = list((prefapmix_output_dir / "stereo").rglob("*_stereo.wav"))
            for stereo_file in stereo_files:
                wb_base = strip_prefix(stereo_file.stem)
                wb_dir = job_output_dir / "wb" / wb_base / "stereo"
                wb_dir.mkdir(parents=True, exist_ok=True)
                run_whisperbite(stereo_file, wb_dir)
                merge_clap_annotations_with_transcript(job_output_dir, wb_dir)
    # --- Soundbiting: Only process (Vocals) trans_out/recv_out files as per flag ---
    elif mode in ("soundbites", "auto"):
        for wav_file in files:
            if wav_file.name.startswith("trans_out") and "(Vocals)" in wav_file.name:
                wb_base = strip_prefix(wav_file.stem)
                wb_dir = job_output_dir / "wb" / wb_base / "right"
                wb_dir.mkdir(parents=True, exist_ok=True)
                run_whisperbite(wav_file, wb_dir)
                merge_clap_annotations_with_transcript(job_output_dir, wb_dir)
            elif process_left_channel and wav_file.name.startswith("recv_out") and "(Vocals)" in wav_file.name:
                wb_base = strip_prefix(wav_file.stem)
                wb_dir = job_output_dir / "wb" / wb_base / "left"
                wb_dir.mkdir(parents=True, exist_ok=True)
                run_whisperbite(wav_file, wb_dir)
                merge_clap_annotations_with_transcript(job_output_dir, wb_dir)
    else:
        logging.info("[mhrpTools] No valid processing mode selected or no matching files found.")

    # When creating the show file:
    if workflow_preset == 'show-edit':
        # Collect all mixed stereo call files
        stereo_dir = output_dir / 'stereo'
        if not stereo_dir.exists():
            logging.error(f"[mhrpTools] No 'stereo' directory found in {output_dir}. Cannot build show.")
            return skipped_files
        call_files = list(stereo_dir.glob('*.wav'))
        if not call_files:
            logging.error(f"[mhrpTools] No mixed stereo call files found in {stereo_dir}. Cannot build show.")
            return skipped_files
        # Sort files chronologically by extracting numeric tokens (e.g., timestamps) from filenames
        def extract_timestamp(f):
            import re
            nums = re.findall(r'\d+', f.stem)
            return int(nums[-1]) if nums else 0
        ordered_call_files = sorted(call_files, key=extract_timestamp)
        # Build concat list: call1, tones, call2, tones, ..., callN (no tones after last call)
        concat_files = []
        for i, call_file in enumerate(ordered_call_files):
            concat_files.append(call_file)
            if i < len(ordered_call_files) - 1:
                concat_files.append(tones_file)
        ffmpeg_concat(concat_files, show_file)
        logging.info(f"[mhrpTools] Show-edit mode: inserted tones.wav only between calls, not at end of calls.")
        # Update metadata to record show-edit mode, call/tone order, tones.wav info
        # ...
        # After generating transcript for each call:
        transcript = ... # get transcript for this call
        call_title = llm_generate_call_title(transcript)
        if call_title:
            call_metadata['llm_title'] = call_title
            # Optionally, rename call folder to include title
            # new_folder = call_folder.parent / f"{call_folder.name}_{call_title}"
            # call_folder.rename(new_folder)
            # call_folder = new_folder
            logging.info(f"[mhrpTools] LLM-generated call title: {call_title}")
    else:
        # Legacy/other modes: previous behavior
        ...

    # After transcript generation for any audio (call or arbitrary):
    if generate_image_prompts:
        image_prompt = llm_generate_image_prompt(transcript)
        if image_prompt:
            with open(call_folder / "image_prompt.txt", "w", encoding="utf-8") as f:
                f.write(image_prompt)
            logging.info(f"[mhrpTools] Saved image prompt to {call_folder / 'image_prompt.txt'}")

    return skipped_files

# At the end of the pipeline, after show concatenation:
def write_show_metadata(show_name, show_file, input_folder, processing_options, calls, output_dir):
    """
    Write a comprehensive show metadata YAML file with all call lineage and transcription data.
    """
    show_metadata = {
        'show': {
            'name': show_name,
            'file': str(show_file),
            'input_folder': str(input_folder),
            'processing_options': processing_options,
            'calls': calls
        }
    }
    metadata_path = output_dir / f"{show_name}_metadata.yaml"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        yaml.dump(show_metadata, f, allow_unicode=True, sort_keys=False)
    logging.info(f"[mhrpTools] Wrote show metadata to {metadata_path}")

# During processing, for each call, collect:
# - original_file, timestamp, outputs, call_start, call_end, tones_appended, transcription (transcript, diarization, word timings, sound events, etc.)
# After concatenating the show, call write_show_metadata with all collected info. 