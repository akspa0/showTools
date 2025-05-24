import os
from pathlib import Path
import soundfile as sf
import mutagen
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3, TIT2, TPE1, TALB, COMM, TDRC, TRCK, TCON, TPOS, USLT, TXXX
import shutil
import json
import glob
import subprocess
import datetime
import re

def sanitize_filename(name, max_length=48):
    name = name.strip().replace(' ', '_')
    name = re.sub(r'[^A-Za-z0-9_\-]', '', name)
    return name[:max_length] or 'untitled'

def wav_to_mp3_youtube_style(wav_path, mp3_path, bitrate='192k'):
    """
    Convert WAV to MP3 with YouTube-style audio processing:
    - Dynamic range compression 
    - Loudness normalization to -14 LUFS
    - True peak limiting to -1.0 dBTP
    - High-frequency clarity enhancement
    """
    import logging
    if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
        print(f"[ERROR] Input WAV file missing or empty: {wav_path}")
        return
    cmd = [
        'ffmpeg', '-y', '-i', str(wav_path),
        # Audio processing chain (YouTube-style)
        '-af', 
        # 1. Dynamic range compression (moderate ratio for broadcast)
        'acompressor=ratio=3:threshold=-18dB:attack=5:release=50:makeup=2dB,'
        # 2. Loudness normalization to -14 LUFS (YouTube standard)
        'loudnorm=I=-14:TP=-1.0:LRA=7:measured_I=-14:measured_LRA=7:measured_TP=-1.0:measured_thresh=-24,'
        # 3. High-frequency clarity boost (subtle)
        'highpass=f=80,treble=g=1.5:f=8000,'
        # 4. Final true peak limiter
        'alimiter=level_in=1:level_out=0.95:limit=0.95:attack=5:release=50',
        # MP3 encoding
        '-codec:a', 'libmp3lame', '-qscale:a', '2', '-b:a', bitrate,
        str(mp3_path)
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        if result.returncode != 0:
            print(f"[WARN] YouTube-style processing failed for {wav_path}, falling back to simple conversion")
            print(f"[FFMPEG STDERR] {result.stderr.decode(errors='ignore')}")
            wav_to_mp3_simple(wav_path, mp3_path, bitrate)
    except subprocess.TimeoutExpired:
        print(f"[ERROR] ffmpeg timed out for {wav_path}, falling back to simple conversion")
        wav_to_mp3_simple(wav_path, mp3_path, bitrate)
    except Exception as e:
        print(f"[ERROR] Exception during ffmpeg for {wav_path}: {e}")
        wav_to_mp3_simple(wav_path, mp3_path, bitrate)

def wav_to_mp3_simple(wav_path, mp3_path, bitrate='192k'):
    """Simple WAV to MP3 conversion without processing"""
    cmd = [
        'ffmpeg', '-y', '-i', str(wav_path), 
        '-codec:a', 'libmp3lame', '-qscale:a', '2', '-b:a', bitrate, 
        str(mp3_path)
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def wav_to_mp3(wav_path, mp3_path, bitrate='192k', youtube_style=True):
    """Convert WAV to MP3 with optional YouTube-style processing"""
    if youtube_style:
        wav_to_mp3_youtube_style(wav_path, mp3_path, bitrate)
    else:
        wav_to_mp3_simple(wav_path, mp3_path, bitrate)

def embed_id3(mp3_path, tags: dict):
    try:
        audio = EasyID3(mp3_path)
    except mutagen.id3.ID3NoHeaderError:
        audio = mutagen.File(mp3_path, easy=True)
        audio.add_tags()
    for k, v in tags.items():
        if v:
            try:
                audio[k] = str(v)
            except Exception:
                pass
    audio.save()

def embed_lineage_id3(mp3_path, original_title, start_time=None, end_time=None):
    try:
        audio = ID3(mp3_path)
    except mutagen.id3.ID3NoHeaderError:
        audio = ID3()
    lineage_comment = f"Source: {original_title}"
    if start_time is not None and end_time is not None:
        lineage_comment += f" | Timestamp: {start_time:.2f}-{end_time:.2f} sec"
    audio.add(COMM(encoding=3, lang='eng', desc='lineage', text=lineage_comment))
    audio.save(mp3_path)

def run_finalization_stage(run_folder: Path, manifest: list):
    finalized_dir = run_folder / 'finalized'
    calls_dir = finalized_dir / 'calls'
    show_dir = finalized_dir / 'show'
    soundbites_dir = finalized_dir / 'soundbites'
    calls_dir.mkdir(parents=True, exist_ok=True)
    show_dir.mkdir(parents=True, exist_ok=True)
    soundbites_dir.mkdir(parents=True, exist_ok=True)
    llm_dir = run_folder / 'llm'
    # --- 1. Calls ---
    call_entries = [e for e in manifest if e.get('stage') == 'remix']
    for entry in call_entries:
        call_id = entry.get('call_id')
        wav_path = Path(entry.get('output_files')[0]) if entry.get('output_files') else None
        call_title = call_id
        call_title_path = llm_dir / call_id / 'call_title.txt'
        if call_title_path.exists():
            with open(call_title_path, 'r', encoding='utf-8') as f:
                lines = [line.strip().strip('"') for line in f.read().splitlines() if line.strip()]
                if lines:
                    call_title = lines[0]
                    # Remove commentary starting with 'This title' (case-insensitive)
                    call_title = re.split(r'(?i)\bthis title\b', call_title)[0].strip()
        sanitized_title = sanitize_filename(call_title)
        mp3_name = sanitized_title + '.mp3'
        mp3_path = calls_dir / mp3_name
        if wav_path and wav_path.exists():
            wav_to_mp3(wav_path, mp3_path)
            embed_id3(mp3_path, {
                'title': call_title,
                'tracknumber': call_id,
                'album': 'Audio Context Calls',
            })
            original_title = entry.get('original_title_for_id3', call_title)
            start_time = entry.get('start_time')
            end_time = entry.get('end_time')
            embed_lineage_id3(mp3_path, original_title, start_time, end_time)
            # --- Copy master transcript to finalized/calls as <sanitized_title>_transcript.txt ---
            soundbites_root = run_folder / 'soundbites'
            transcript_src = soundbites_root / sanitized_title / 'master_transcript.txt'
            transcript_dst = calls_dir / f'{sanitized_title}_transcript.txt'
            if transcript_src.exists():
                shutil.copy2(transcript_src, transcript_dst)
            else:
                print(f"[WARN] Master transcript not found for call {sanitized_title}: {transcript_src}")
    # --- 2. Show ---
    show_wav = run_folder / 'show' / 'show.wav'
    show_json = run_folder / 'show' / 'show.json'
    show_title = 'completed-show'
    show_desc = ''
    show_title_path = llm_dir / 'show_title.txt'
    show_desc_path = llm_dir / 'show_description.txt'
    if show_title_path.exists():
        with open(show_title_path, 'r', encoding='utf-8') as f:
            t = f.read().strip().strip('"')
            if t:
                show_title = sanitize_filename(t)
    if show_desc_path.exists():
        with open(show_desc_path, 'r', encoding='utf-8') as f:
            show_desc = f.read().strip()
    show_mp3 = show_dir / (show_title + '.mp3')
    if show_wav.exists():
        wav_to_mp3(show_wav, show_mp3)
        embed_id3(show_mp3, {
            'title': show_title,
            'album': 'Audio Context Show',
            'comment': show_desc,
        })
        # Only use show-level metadata for the show MP3; do not use 'entry' from previous loop
        original_title = show_title or 'unknown_source'
        # Defensive: do not use start_time/end_time from 'entry' (not defined here)
        embed_lineage_id3(show_mp3, original_title)
    # --- 3. Show TXT ---
    def format_hms(seconds):
        return str(datetime.timedelta(seconds=int(seconds)))
    show_txt = show_dir / (show_title + '.txt')
    timeline = []
    call_titles = []
    if show_json.exists():
        with open(show_json, 'r', encoding='utf-8') as f:
            timeline = json.load(f)
    with open(show_txt, 'w', encoding='utf-8') as f:
        if show_title:
            f.write(f"Show Title: {show_title}\n")
        if show_desc:
            f.write(f"Show Description: {show_desc}\n\n")
        f.write("Call Timeline:\n")
        for entry in timeline:
            if 'call_title' in entry:
                call_title = entry['call_title']
                call_id = entry.get('call_id')
                call_title_path = llm_dir / call_id / 'call_title.txt'
                if call_title_path.exists():
                    with open(call_title_path, 'r', encoding='utf-8') as tf:
                        lines = [line.strip().strip('"') for line in tf.read().splitlines() if line.strip()]
                        if lines:
                            call_title = lines[0]
                            call_title = re.split(r'(?i)\bthis title\b', call_title)[0].strip()
                call_titles.append(call_title)
                start = format_hms(entry['start'])
                end = format_hms(entry['end'])
                f.write(f"🎙️ {call_title} ({start} - {end})\n")
            elif 'tones' in entry:
                start = format_hms(entry['start'])
                end = format_hms(entry['end'])
                f.write(f"🔔 [Tones] ({start} - {end})\n")
    # --- 3b. Show Notes LLM Task ---
    # Compose prompt for show notes
    show_notes_prompt = (
        "Given the following list of call titles, write a short, whimsical, and absurdist set of show notes that "
        "captures the comedic and surreal nature of the calls. Be cheery, playful, and concise. Do not repeat the call titles verbatim, "
        "but reference the themes and energy of the show.\n\nCall Titles:\n" + '\n'.join(f'- {title}' for title in call_titles)
    )
    show_notes_txt = show_dir / 'show-notes.txt'
    # Run LLM for show notes (reuse LLM config from orchestrator if available)
    llm_config_path = run_folder / 'workflows' / 'llm_tasks.json'
    llm_config = None
    if llm_config_path.exists():
        with open(llm_config_path, 'r', encoding='utf-8') as f:
            llm_config = json.load(f)
    if llm_config:
        import requests, random, hashlib
        base_url = llm_config.get('lm_studio_base_url', 'http://localhost:1234/v1')
        api_key = llm_config.get('lm_studio_api_key', 'lm-studio')
        model_id = llm_config.get('lm_studio_model_identifier', 'llama-3.1-8b-supernova-etherealhermes')
        temperature = llm_config.get('lm_studio_temperature', 0.5)
        max_tokens = llm_config.get('lm_studio_max_tokens', 250)
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        seed_input = f"show_notes_{show_title}"
        seed = int(hashlib.sha256(seed_input.encode()).hexdigest(), 16) % (2**32)
        data = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": show_notes_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed
        }
        try:
            response = requests.post(f"{base_url}/chat/completions", headers=headers, data=json.dumps(data), timeout=60)
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                with open(show_notes_txt, 'w', encoding='utf-8') as nf:
                    nf.write(content)
                    nf.write("\n\n---\n\n")
                    nf.write("Call Timeline:\n")
                    for entry in timeline:
                        if 'call_title' in entry:
                            call_title = entry['call_title']
                            call_id = entry.get('call_id')
                            call_title_path = llm_dir / call_id / 'call_title.txt'
                            if call_title_path.exists():
                                with open(call_title_path, 'r', encoding='utf-8') as tf:
                                    lines = [line.strip().strip('"') for line in tf.read().splitlines() if line.strip()]
                                    if lines:
                                        call_title = lines[0]
                            start = format_hms(entry['start'])
                            end = format_hms(entry['end'])
                            nf.write(f"🎙️ {call_title} ({start} - {end})\n")
                        elif 'tones' in entry:
                            start = format_hms(entry['start'])
                            end = format_hms(entry['end'])
                            nf.write(f"🔔 [Tones] ({start} - {end})\n")
        except Exception as e:
            with open(show_notes_txt, 'w', encoding='utf-8') as nf:
                nf.write(f"[ERROR] Failed to generate show notes: {e}\n")
    # --- 3c. Safe-for-Work Call Titles LLM Task ---
    for entry in timeline:
        if 'call_title' in entry:
            call_id = entry.get('call_id')
            call_title_path = llm_dir / call_id / 'call_title.txt'
            transcript_path = soundbites_dir / call_id / 'master_transcript.txt'
            sfw_title_path = llm_dir / call_id / 'call_title_sfw.txt'
            if call_title_path.exists() and transcript_path.exists():
                with open(call_title_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip().strip('"') for line in f.read().splitlines() if line.strip()]
                    if lines:
                        orig_title = lines[0]
                        orig_title = re.split(r'(?i)\bthis title\b', orig_title)[0].strip()
                with open(transcript_path, 'r', encoding='utf-8') as tf:
                    transcript = tf.read().strip()
                sfw_prompt = (
                    f"Given the following call title and transcript, generate a safe-for-work, family-friendly version of the call title. "
                    f"Do not include any PII or inappropriate language.\n\nCall Title: {orig_title}\n\nTranscript:\n{transcript}"
                )
                if llm_config:
                    import requests, random, hashlib
                    base_url = llm_config.get('lm_studio_base_url', 'http://localhost:1234/v1')
                    api_key = llm_config.get('lm_studio_api_key', 'lm-studio')
                    model_id = llm_config.get('lm_studio_model_identifier', 'llama-3.1-8b-supernova-etherealhermes')
                    temperature = llm_config.get('lm_studio_temperature', 0.5)
                    max_tokens = llm_config.get('lm_studio_max_tokens', 100)
                    headers = {
                        'Authorization': f'Bearer {api_key}',
                        'Content-Type': 'application/json'
                    }
                    seed_input = f"sfw_title_{call_id}"
                    seed = int(hashlib.sha256(seed_input.encode()).hexdigest(), 16) % (2**32)
                    data = {
                        "model": model_id,
                        "messages": [
                            {"role": "user", "content": sfw_prompt}
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "seed": seed
                    }
                    try:
                        response = requests.post(f"{base_url}/chat/completions", headers=headers, data=json.dumps(data), timeout=60)
                        if response.status_code == 200:
                            result = response.json()
                            content = result['choices'][0]['message']['content'].strip()
                            # Remove commentary starting with 'This title' (case-insensitive)
                            content = re.split(r'(?i)\bthis title\b', content)[0].strip()
                            with open(sfw_title_path, 'w', encoding='utf-8') as sfwf:
                                sfwf.write(content)
                    except Exception as e:
                        with open(sfw_title_path, 'w', encoding='utf-8') as sfwf:
                            sfwf.write(f"[ERROR] Failed to generate SFW call title: {e}\n")
    # --- 3d. Complete SFW Show TXT ---
    complete_sfw_txt = show_dir / 'complete-show-sfw.txt'
    with open(complete_sfw_txt, 'w', encoding='utf-8') as f:
        f.write(f"Show Title: {show_title} (Safe for Work)\n")
        if show_desc:
            f.write(f"Show Description: {show_desc}\n\n")
        f.write("Call Timeline:\n")
        for entry in timeline:
            if 'call_title' in entry:
                call_id = entry.get('call_id')
                sfw_title_path = llm_dir / call_id / 'call_title_sfw.txt'
                call_title = None
                if sfw_title_path.exists():
                    with open(sfw_title_path, 'r', encoding='utf-8') as sfwf:
                        lines = [line.strip().strip('"') for line in sfwf.read().splitlines() if line.strip()]
                        if lines:
                            call_title = lines[0]
                            call_title = re.split(r'(?i)\bthis title\b', call_title)[0].strip()
                if not call_title:
                    # fallback to original call title
                    call_title_path = llm_dir / call_id / 'call_title.txt'
                    if call_title_path.exists():
                        with open(call_title_path, 'r', encoding='utf-8') as tf:
                            lines = [line.strip().strip('"') for line in tf.read().splitlines() if line.strip()]
                            if lines:
                                call_title = lines[0]
                                call_title = re.split(r'(?i)\bthis title\b', call_title)[0].strip()
                start = format_hms(entry['start'])
                end = format_hms(entry['end'])
                f.write(f"🎙️ {call_title} ({start} - {end})\n")
            elif 'tones' in entry:
                start = format_hms(entry['start'])
                end = format_hms(entry['end'])
                f.write(f"🔔 [Tones] ({start} - {end})\n")
    # --- 3e. Append LLM Show Notes to Show TXT and SFW Show TXT ---
    def append_llm_show_notes(call_titles, out_path):
        show_notes_prompt = (
            "Given the following list of call titles, write a concise, engaging set of show notes that summarizes the show for listeners. "
            "Be playful, creative, and avoid repeating the call titles verbatim.\n\nCall Titles:\n" + '\n'.join(f'- {title}' for title in call_titles)
        )
        if llm_config:
            import requests, random, hashlib
            base_url = llm_config.get('lm_studio_base_url', 'http://localhost:1234/v1')
            api_key = llm_config.get('lm_studio_api_key', 'lm-studio')
            model_id = llm_config.get('lm_studio_model_identifier', 'llama-3.1-8b-supernova-etherealhermes')
            temperature = llm_config.get('lm_studio_temperature', 0.5)
            max_tokens = llm_config.get('lm_studio_max_tokens', 250)
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            seed_input = f"show_notes_{out_path.name}"
            seed = int(hashlib.sha256(seed_input.encode()).hexdigest(), 16) % (2**32)
            data = {
                "model": model_id,
                "messages": [
                    {"role": "user", "content": show_notes_prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed
            }
            try:
                response = requests.post(f"{base_url}/chat/completions", headers=headers, data=json.dumps(data), timeout=60)
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content'].strip()
                    with open(out_path, 'a', encoding='utf-8') as f:
                        f.write("\n\nShow Notes:\n")
                        f.write(content)
                        f.write("\n")
            except Exception as e:
                with open(out_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n[ERROR] Failed to generate show notes: {e}\n")
    # Append to complete-show.txt (original call titles)
    append_llm_show_notes(call_titles, show_txt)
    # Append to complete-show-sfw.txt (SFW call titles if available, else fallback)
    sfw_call_titles = []
    for entry in timeline:
        if 'call_title' in entry:
            call_id = entry.get('call_id')
            sfw_title_path = llm_dir / call_id / 'call_title_sfw.txt'
            call_title = None
            if sfw_title_path.exists():
                with open(sfw_title_path, 'r', encoding='utf-8') as sfwf:
                    lines = [line.strip().strip('"') for line in sfwf.read().splitlines() if line.strip()]
                    if lines:
                        call_title = lines[0]
                        call_title = re.split(r'(?i)\bthis title\b', call_title)[0].strip()
            if not call_title:
                call_title_path = llm_dir / call_id / 'call_title.txt'
                if call_title_path.exists():
                    with open(call_title_path, 'r', encoding='utf-8') as tf:
                        lines = [line.strip().strip('"') for line in tf.read().splitlines() if line.strip()]
                        if lines:
                            call_title = lines[0]
                            call_title = re.split(r'(?i)\bthis title\b', call_title)[0].strip()
            sfw_call_titles.append(call_title)
    append_llm_show_notes(sfw_call_titles, complete_sfw_txt)
    # --- 4. Soundbites ---
    soundbites_root = run_folder / 'soundbites'
    finalized_soundbites_root = finalized_dir / 'soundbites'
    clap_root = run_folder / 'clap'
    # --- Build mapping from call_id to sanitized soundbites folder name ---
    call_id_to_folder = {}
    for folder in soundbites_root.iterdir():
        if not folder.is_dir():
            continue
        # Skip conversation/out- folders
        if 'conversation' in folder.name or folder.name.startswith('out-'):
            print(f"[WARN] Skipping conversation/out- folder in soundbites: {folder}")
            continue
        # Try to find call_id from manifest or folder contents
        if folder.name[:4].isdigit():
            call_id = folder.name[:4]
            call_id_to_folder[call_id] = folder.name
        else:
            for file in folder.glob('*_master_transcript.txt'):
                parts = file.name.split('_')
                if parts and parts[0][:4].isdigit():
                    call_id = parts[0][:4]
                    call_id_to_folder[call_id] = folder.name
    # For each valid call, copy to finalized/soundbites/<call_title>/
    for call_id in sorted(call_id_to_folder.keys()):
        folder_name = call_id_to_folder[call_id]
        call_dir = soundbites_root / folder_name
        if not call_dir.exists():
            print(f"[WARN] Soundbites folder missing for call_id {call_id}: {call_dir}")
            continue
        # Defensive: skip if this is a conversation/out- folder
        if 'conversation' in folder_name or folder_name.startswith('out-'):
            print(f"[WARN] Skipping conversation/out- folder in finalization: {call_dir}")
            continue
        # Determine sanitized call title
        call_title = call_id
        call_title_path = llm_dir / call_id / 'call_title.txt'
        if call_title_path.exists():
            with open(call_title_path, 'r', encoding='utf-8') as f:
                lines = [line.strip().strip('"') for line in f.read().splitlines() if line.strip()]
                if lines:
                    call_title = lines[0]
                    call_title = re.split(r'(?i)\bthis title\b', call_title)[0].strip()
        sanitized_title = sanitize_filename(call_title)
        finalized_call_dir = finalized_soundbites_root / sanitized_title
        # Copy all files and subfolders from call_dir to finalized_call_dir
        for root, dirs, files in os.walk(call_dir):
            rel_root = Path(root).relative_to(call_dir)
            target_root = finalized_call_dir / rel_root
            target_root.mkdir(parents=True, exist_ok=True)
            for file in files:
                src_file = Path(root) / file
                dst_file = target_root / file
                try:
                    shutil.copy2(src_file, dst_file)
                    print(f"[INFO] Copied {src_file} -> {dst_file}")
                except Exception as e:
                    print(f"[WARN] Failed to copy {src_file} -> {dst_file}: {e}")
        # Copy master transcript (.txt and .json) if present
        master_txt = call_dir / 'master_transcript.txt'
        master_json = call_dir / 'master_transcript.json'
        if master_txt.exists():
            try:
                shutil.copy2(master_txt, finalized_call_dir / 'master_transcript.txt')
                print(f"[INFO] Copied master transcript for {sanitized_title}")
            except Exception as e:
                print(f"[WARN] Failed to copy master transcript for {sanitized_title}: {e}")
        else:
            print(f"[WARN] Master transcript missing for {sanitized_title}: {master_txt}")
        if master_json.exists():
            try:
                shutil.copy2(master_json, finalized_call_dir / 'master_transcript.json')
            except Exception as e:
                print(f"[WARN] Failed to copy master transcript JSON for {sanitized_title}: {e}")
        # Copy per-speaker transcripts if present
        for channel_dir in (call_dir.iterdir() if call_dir.exists() else []):
            if not channel_dir.is_dir():
                continue
            for speaker_dir in channel_dir.iterdir():
                if not speaker_dir.is_dir():
                    continue
                for txt_file in speaker_dir.glob('*.txt'):
                    try:
                        rel_path = txt_file.relative_to(call_dir)
                        dst_file = finalized_call_dir / rel_path
                        dst_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(txt_file, dst_file)
                        print(f"[INFO] Copied per-speaker transcript {txt_file} -> {dst_file}")
                    except Exception as e:
                        print(f"[WARN] Failed to copy per-speaker transcript {txt_file}: {e}")
        # Gather CLAP events for this call (confidence >= 0.9)
        clap_events = []
        master_transcript_events = []
        clap_call_dir = clap_root / call_dir.name
        if clap_call_dir.exists():
            for clap_file in clap_call_dir.glob('*.json'):
                with open(clap_file, 'r', encoding='utf-8') as f:
                    try:
                        clap_data = json.load(f)
                        for event in clap_data:
                            conf = event.get('confidence', 0)
                            if conf >= 0.9:
                                start = event.get('start_time', event.get('start', 0))
                                label = event.get('prompt', event.get('label', 'unknown'))
                                master_transcript_events.append({
                                    'type': 'clap',
                                    'timestamp': start,
                                    'text': f"[Annotation][{start:.2f}]: {label}"
                                })
                    except Exception:
                        continue
        # Gather utterances and finalize soundbites (including single-file jobs)
        for channel_dir in call_dir.iterdir():
            if not channel_dir.is_dir():
                continue
            channel_label = channel_dir.name.upper().replace('-VOCALS', '')
            for speaker_dir in channel_dir.iterdir():
                if not speaker_dir.is_dir():
                    continue
                speaker_label = speaker_dir.name.upper()
                out_speaker_dir = soundbites_dir / call_dir.name / channel_dir.name / speaker_dir.name
                out_speaker_dir.mkdir(parents=True, exist_ok=True)
                for wav_file in speaker_dir.glob('*.wav'):
                    base = wav_file.stem
                    txt_file = wav_file.with_suffix('.txt')
                    if not txt_file.exists():
                        continue
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        transcript = f.read().strip()
                    if not transcript:
                        continue
                    parts = base.split('-')
                    try:
                        start = float(parts[1])/100 if len(parts) > 1 else 0
                        end = float(parts[2])/100 if len(parts) > 2 else 0
                    except Exception:
                        start = 0
                        end = 0
                    mp3_name = sanitize_filename(base) + '.mp3'
                    mp3_path = out_speaker_dir / mp3_name
                    wav_to_mp3(wav_file, mp3_path)
                    embed_id3(mp3_path, {
                        'title': transcript,
                        'album': 'Audio Context Soundbites',
                        'comment': f"{channel_dir.name} {speaker_dir.name} from {call_dir.name}; file: {wav_file.name}; start: {start:.2f}; end: {end:.2f}",
                    })
                    out_txt_path = out_speaker_dir / (sanitize_filename(base) + '.txt')
                    shutil.copy2(txt_file, out_txt_path)
                    # Add to master transcript (utterance: no type field in JSON)
                    line = f"[{channel_label}][{speaker_label}][{start:.2f}]: {transcript}"
                    master_transcript_events.append({
                        'timestamp': start,
                        'text': line
                    })
                    # Retrieve original_title_for_id3 from manifest/job data
                    original_title = None
                    if 'entry' in locals() and entry is not None:
                        original_title = entry.get('original_title_for_id3', transcript)
                    else:
                        # Fallback: use transcript or a default value
                        original_title = transcript or 'unknown_source'
                    # Retrieve start/end time for the soundbite (if available)
                    start_time = start
                    end_time = end
                    embed_lineage_id3(mp3_path, original_title, start_time, end_time) 