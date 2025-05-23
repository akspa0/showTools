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
            # Retrieve original_title_for_id3 from manifest/job data
            original_title = None
            if 'entry' in locals() and entry is not None:
                original_title = entry.get('original_title_for_id3', call_title)
            else:
                # Fallback: use call_title or a default value
                original_title = call_title or 'unknown_source'
            # Retrieve start/end time for the soundbite (if available)
            start_time = entry.get('start_time')
            end_time = entry.get('end_time')
            embed_lineage_id3(mp3_path, original_title, start_time, end_time)
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
        # Retrieve original_title_for_id3 from manifest/job data
        original_title = None
        if 'entry' in locals() and entry is not None:
            original_title = entry.get('original_title_for_id3', show_title)
        else:
            # Fallback: use show_title or a default value
            original_title = show_title or 'unknown_source'
        # Retrieve start/end time for the soundbite (if available)
        start_time = entry.get('start_time')
        end_time = entry.get('end_time')
        embed_lineage_id3(show_mp3, original_title, start_time, end_time)
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
    soundbites_root = run_folder / 'soundbites'
    clap_root = run_folder / 'clap'
    for call_dir in soundbites_root.iterdir():
        if not call_dir.is_dir():
            continue
        master_transcript_events = []
        # Gather CLAP events for this call (confidence >= 0.9)
        clap_events = []
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
        # Sort all events by timestamp
        master_transcript_events_sorted = sorted(master_transcript_events, key=lambda x: x['timestamp'])
        # Write .txt
        master_txt_path = soundbites_dir / call_dir.name / 'master_transcript.txt'
        master_txt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(master_txt_path, 'w', encoding='utf-8') as f:
            for ev in master_transcript_events_sorted:
                f.write(ev['text'] + '\n')
        # Write .json
        master_json_path = soundbites_dir / call_dir.name / 'master_transcript.json'
        master_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(master_json_path, 'w', encoding='utf-8') as f:
            json.dump(master_transcript_events_sorted, f, indent=2, ensure_ascii=False) 