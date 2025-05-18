import os
import shutil
import json
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from audio_processor import AudioProcessor
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3, TXXX, COMM
import subprocess
import zipfile
import string
from pydub import AudioSegment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sanitize_name(name: str, fallback: str) -> str:
    # Remove punctuation, limit to 8 words, fallback if empty
    name = re.sub(r'[^\w\s-]', '', name)
    words = name.split()
    if not words:
        return fallback
    return '_'.join(words[:8])

def delete_and_rebuild_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def find_call_folders(root: Path) -> List[Path]:
    call_folders = []
    for dirpath, dirnames, filenames in os.walk(root):
        if any(f.endswith('.wav') or f.endswith('.mp3') for f in filenames):
            call_folders.append(Path(dirpath))
    return call_folders

def get_llm_name(call_folder: Path, call_id: str) -> str:
    name_file = next(call_folder.glob('*_suggested_name.txt'), None)
    if name_file and name_file.exists():
        with open(name_file, 'r', encoding='utf-8') as f:
            name = f.read().strip()
            return sanitize_name(name, call_id)
    return call_id

def get_timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def tag_mp3(mp3_path: Path, call_name: str, synopsis: str, orig_meta: dict):
    try:
        try:
            audio = EasyID3(str(mp3_path))
        except Exception:
            audio = ID3()
        audio['title'] = call_name
        if synopsis:
            audio['comment'] = synopsis
        if orig_meta:
            comment_str = json.dumps(orig_meta, ensure_ascii=False)
            audio['COMM'] = comment_str[:1024]
        audio.save(str(mp3_path))
    except Exception as e:
        logging.warning(f"Failed to tag MP3 {mp3_path}: {e}")

def copy_transcripts(call_folder: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in call_folder.glob('*.txt'):
        shutil.copy2(f, out_dir / f.name)
    for f in call_folder.glob('*.json'):
        shutil.copy2(f, out_dir / f.name)

def copy_llm_and_metadata(call_folder: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in call_folder.glob('*.txt'):
        shutil.copy2(f, out_dir / f.name)
    for f in call_folder.glob('*.json'):
        shutil.copy2(f, out_dir / f.name)

def convert_and_copy_soundbites(call_folder: Path, out_dir: Path, call_name: str, orig_meta: dict, synopsis: str):
    for speaker_dir in [d for d in call_folder.iterdir() if d.is_dir() and d.name.startswith('S')]:
        speaker_out = out_dir / speaker_dir.name
        speaker_out.mkdir(parents=True, exist_ok=True)
        for wav_file in speaker_dir.glob('*.wav'):
            mp3_path = speaker_out / (wav_file.stem + '.mp3')
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-i', str(wav_file),
                '-codec:a', 'libmp3lame', '-qscale:a', '2', str(mp3_path)
            ]
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0 and mp3_path.exists():
                tag_mp3(mp3_path, call_name, synopsis, orig_meta)
            else:
                logging.warning(f"Failed to convert soundbite {wav_file} to MP3: {result.stderr}")
            # Copy corresponding .txt if exists
            txt_file = wav_file.with_suffix('.txt')
            if txt_file.exists():
                shutil.copy2(txt_file, speaker_out / txt_file.name)

def build_final_output(input_root: Path, output_root: Path):
    delete_and_rebuild_dir(output_root)
    trans_dir = output_root / 'Transcriptions'
    sound_dir = output_root / 'Soundbites'
    meta_dir = output_root / 'Metadata'
    bad_calls = []
    all_metadata = {}
    processor = AudioProcessor()
    call_folders = find_call_folders(input_root)
    for call_folder in call_folders:
        call_id = call_folder.name
        call_name = get_llm_name(call_folder, call_id)
        timestamp = get_timestamp()
        out_mp3_name = f"{call_name}_{timestamp}.mp3"
        out_mp3_path = output_root / out_mp3_name
        # Find stems
        vocal_stem = next(call_folder.rglob('*vocals_normalized.wav'), None)
        instrumental_stem = next(call_folder.rglob('*instrumental_normalized.wav'), None)
        if not vocal_stem or not instrumental_stem:
            bad_calls.append({'call_id': call_id, 'reason': 'Missing stems'})
            continue
        # Mix and compress
        mixed_wav = output_root / f"{call_name}_{timestamp}_mixed.wav"
        try:
            # Use pydub to mix
            if not vocal_stem.exists() or not instrumental_stem.exists():
                bad_calls.append({'call_id': call_id, 'reason': 'Missing stem files'})
                continue
            vocals = AudioSegment.from_wav(vocal_stem)
            instrumental = AudioSegment.from_wav(instrumental_stem) - 6  # Reduce instrumental by ~50% (6dB)
            mixed = vocals.overlay(instrumental)
            mixed.export(mixed_wav, format="wav")
            if not mixed_wav.exists() or mixed_wav.stat().st_size == 0:
                bad_calls.append({'call_id': call_id, 'reason': 'Mixing produced empty file'})
                continue
        except Exception as e:
            bad_calls.append({'call_id': call_id, 'reason': f'Mixing exception: {e}'})
            continue
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', str(mixed_wav),
            '-codec:a', 'libmp3lame', '-qscale:a', '2', str(out_mp3_path)
        ]
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not out_mp3_path.exists():
            bad_calls.append({'call_id': call_id, 'reason': 'MP3 conversion failed'})
            continue
        # Tag MP3
        synopsis_file = next(call_folder.glob('*_combined_call_summary.txt'), None)
        synopsis = synopsis_file.read_text().strip() if synopsis_file and synopsis_file.exists() else ''
        orig_meta = {}
        for meta_file in call_folder.glob('*.json'):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    orig_meta.update(meta)
            except Exception:
                continue
        tag_mp3(out_mp3_path, call_name, synopsis, orig_meta)
        # Copy transcripts
        call_trans_dir = trans_dir / call_name
        copy_transcripts(call_folder, call_trans_dir)
        # Copy LLM outputs and metadata
        call_meta_dir = meta_dir / call_name
        copy_llm_and_metadata(call_folder, call_meta_dir)
        # Convert/copy soundbites
        call_sound_dir = sound_dir / call_name
        convert_and_copy_soundbites(call_folder, call_sound_dir, call_name, orig_meta, synopsis)
        # Metadata
        all_metadata[call_name] = {
            'call_id': call_id,
            'mp3': str(out_mp3_path.relative_to(output_root)),
            'transcripts': [str(f.relative_to(output_root)) for f in call_trans_dir.glob('*')],
            'soundbites': [str(f.relative_to(output_root)) for f in call_sound_dir.rglob('*.mp3')],
            'metadata': [str(f.relative_to(output_root)) for f in call_meta_dir.glob('*')],
        }
        # Clean up temp wav
        if mixed_wav.exists():
            mixed_wav.unlink()
    # Write metadata and bad calls
    with open(output_root / 'final_output_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    with open(output_root / 'bad_calls.txt', 'w', encoding='utf-8') as f:
        for bad in bad_calls:
            f.write(json.dumps(bad) + '\n')
    # Zip output
    zip_path = output_root / 'final_output.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_root):
            for file in files:
                file_path = os.path.join(root, file)
                if file_path == str(zip_path):
                    continue
                arcname = os.path.relpath(file_path, output_root)
                zipf.write(file_path, arcname)
    logging.info(f"Final output built at {output_root}, zipped as {zip_path}")

def cleanup_small_folders(output_root: Path, min_size_kb: int = 192):
    for call_dir in output_root.iterdir():
        if call_dir.is_dir() and call_dir.name not in ['Transcriptions', 'Soundbites', 'Metadata']:
            total_size = sum(f.stat().st_size for f in call_dir.rglob('*') if f.is_file())
            if total_size < min_size_kb * 1024:
                shutil.rmtree(call_dir)
                logging.info(f"Removed small output folder: {call_dir} (size: {total_size/1024:.1f} KB)")

def sanitize_call_name(name: str) -> str:
    name = name.translate(str.maketrans('', '', string.punctuation))
    words = name.strip().split()
    sanitized = '_'.join(words[:8])
    return sanitized or None

def rename_folders_with_call_names(output_root: Path):
    existing_names = set()
    for call_dir in list(output_root.iterdir()):
        if not call_dir.is_dir() or call_dir.name in ['Transcriptions', 'Soundbites', 'Metadata']:
            continue
        name_file = next(call_dir.glob('*_suggested_name.txt'), None)
        if name_file and name_file.exists():
            with open(name_file, 'r', encoding='utf-8') as f:
                call_name = f.read().strip()
            sanitized = sanitize_call_name(call_name) if call_name else None
        else:
            sanitized = None
        if not sanitized:
            sanitized = call_dir.name
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
    if len(sys.argv) < 2:
        print("Usage: python final_output_builder.py <input_root> [<output_root>]")
        exit(1)
    input_root = Path(sys.argv[1]).resolve()
    output_root = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else (input_root / '05_final_output')
    build_final_output(input_root, output_root)
    cleanup_small_folders(output_root)
    rename_folders_with_call_names(output_root) 