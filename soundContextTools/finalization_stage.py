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

def sanitize_filename(name, max_length=48):
    import re
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
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"[WARN] YouTube-style processing failed for {wav_path}, falling back to simple conversion")
        # Fallback to simple conversion
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
        wav_path = Path(entry['output_files'][0]) if entry.get('output_files') else None
        call_title = call_id
        call_title_path = llm_dir / call_id / 'call_title.txt'
        if call_title_path.exists():
            with open(call_title_path, 'r', encoding='utf-8') as f:
                title = f.read().strip().strip('"')
                if title:
                    call_title = title
        mp3_name = sanitize_filename(call_title) + '.mp3'
        mp3_path = calls_dir / mp3_name
        if wav_path and wav_path.exists():
            wav_to_mp3(wav_path, mp3_path)
            embed_id3(mp3_path, {
                'title': call_title,
                'tracknumber': call_id,
                'album': 'Audio Context Calls',
            })
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
    # --- 3. Show TXT ---
    show_txt = show_dir / (show_title + '.txt')
    timeline = []
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
                f.write(f"- {entry['call_title']} ({entry['start']:.2f}s - {entry['end']:.2f}s)\n")
            elif 'tones' in entry:
                f.write(f"- [Tones] ({entry['start']:.2f}s - {entry['end']:.2f}s)\n")
    # --- 4. Soundbites ---
    soundbite_entries = [e for e in manifest if e.get('stage') == 'final_soundbite']
    for entry in soundbite_entries:
        wav_path = Path(entry.get('soundbite_wav')) if entry.get('soundbite_wav') else None
        transcript = entry.get('transcript')
        call_id = entry.get('call_id')
        channel = entry.get('channel')
        speaker = entry.get('speaker')
        start = entry.get('start')
        end = entry.get('end')
        input_name = entry.get('input_files', [None])[0]
        if input_name:
            input_name = Path(input_name).stem
        # Defensive: skip if start or end is None
        if start is None or end is None:
            print(f"[WARN] Skipping soundbite with missing start/end: {entry}")
            continue
        mp3_name = f"{call_id}_{channel}_{speaker}_{int(start*100):06d}_{int(end*100):06d}.mp3"
        mp3_path = soundbites_dir / mp3_name
        if wav_path and wav_path.exists():
            wav_to_mp3(wav_path, mp3_path)
            embed_id3(mp3_path, {
                'title': transcript or mp3_name,
                'tracknumber': call_id,
                'album': 'Audio Context Soundbites',
                'comment': f"{channel} {speaker} {start:.2f}-{end:.2f}s from {input_name}",
            }) 