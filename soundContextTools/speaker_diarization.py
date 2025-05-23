import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from huggingface_hub import login
from pyannote.audio import Pipeline
# Optimization imports
import torch
import torchaudio
from pyannote.audio.pipelines.utils.hook import ProgressHook

# --- CONFIGURATION ---
MODEL_ID = "pyannote/speaker-diarization-3.1"
SAMPLE_RATE = 16000  # For inference only, not for output


def diarize_file(
    input_wav: str,
    output_dir: str,
    hf_token: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> Dict:
    """
    Run speaker diarization on a single WAV file and save RTTM and JSON outputs.
    Uses robust file-based approach for stability and compatibility.
    Returns dict with RTTM/JSON paths and segment metadata, or error info.
    """
    try:
        if hf_token:
            login(token=hf_token, add_to_git_credential=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline = Pipeline.from_pretrained(MODEL_ID, use_auth_token=hf_token)
        pipeline.to(device)
        pipeline_kwargs = {}
        if min_speakers is not None:
            pipeline_kwargs['min_speakers'] = min_speakers
        if max_speakers is not None:
            pipeline_kwargs['max_speakers'] = max_speakers
        # Run diarization on file path
        diarization = pipeline(str(Path(input_wav).resolve()), **pipeline_kwargs)
        # Prepare output paths
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        base = Path(input_wav).stem
        rttm_path = os.path.join(output_dir, f"{base}.rttm")
        json_path = os.path.join(output_dir, f"{base}.json")
        # Write RTTM manually
        with open(rttm_path, 'w', encoding='utf-8') as f_rttm:
            file_id = base
            segments = []
            for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                start_time = float(turn.start)
                end_time = float(turn.end)
                duration = end_time - start_time
                f_rttm.write(f"SPEAKER {file_id} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {speaker_label} <NA> <NA>\n")
                segments.append({
                    "start": start_time,
                    "end": end_time,
                    "speaker": speaker_label
                })
        # Write JSON
        with open(json_path, "w", encoding="utf-8") as f_json:
            json.dump(segments, f_json, indent=2)
        return {
            "rttm": rttm_path,
            "json": json_path,
            "segments": segments,
            "error": None
        }
    except Exception as e:
        print(f"[ERROR] Diarization failed for {input_wav}: {e}")
        return {
            "rttm": None,
            "json": None,
            "segments": [],
            "error": str(e)
        }


def batch_diarize(
    separated_root: str,
    diarized_root: str,
    hf_token: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    progress: bool = True,
) -> List[Dict]:
    """
    Batch process all *-vocals.wav files and *-conversation.wav files in separated/<call id>/ and output to diarized/<call id>/.
    Returns a list of output metadata dicts.
    """
    results = []
    for call_id in os.listdir(separated_root):
        call_dir = os.path.join(separated_root, call_id)
        if not os.path.isdir(call_dir):
            continue
        diarized_dir = os.path.join(diarized_root, call_id)
        Path(diarized_dir).mkdir(parents=True, exist_ok=True)
        
        # Find both traditional vocal files and conversation files
        audio_files = []
        # Traditional pattern: *-vocals.wav (from separation)
        vocals = [f for f in os.listdir(call_dir) if f.endswith("-vocals.wav")]
        audio_files.extend(vocals)
        # Single file pattern: *-conversation.wav (from single file processing)
        conversations = [f for f in os.listdir(call_dir) if f.endswith("-conversation.wav")]
        audio_files.extend(conversations)
        
        for audio_file in tqdm(audio_files, desc=f"Diarizing {call_id}", disable=not progress):
            input_wav = os.path.join(call_dir, audio_file)
            try:
                out = diarize_file(
                    input_wav,
                    diarized_dir,
                    hf_token=hf_token,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )
                out["call_id"] = call_id
                out["input_name"] = audio_file
                results.append(out)
            except Exception as e:
                # No PII in error logs
                print(f"[WARN] Diarization failed for {call_id}/{audio_file}: {e}")
    return results


def segment_speakers_from_diarization(
    diarized_root: str,
    separated_root: str,
    speakers_root: str,
    progress: bool = True,
) -> list:
    """
    For each diarized/<call id>/<channel>-vocals.json, extract speaker segments from separated/<call id>/<channel>-vocals.wav.
    Save each segment as speakers/<call id>/<channel>/<speaker>/<index>-<start>-<end>.wav.
    Output a JSON and TXT log per call/channel with segment metadata and [SXX][start-end] lines.
    Returns a list of all segment metadata dicts.
    """
    import soundfile as sf
    import numpy as np
    results = []
    for call_id in os.listdir(diarized_root):
        call_dir = os.path.join(diarized_root, call_id)
        if not os.path.isdir(call_dir):
            continue
        for f in os.listdir(call_dir):
            if not f.endswith(".json"):
                continue
            diar_json_path = os.path.join(call_dir, f)
            with open(diar_json_path, "r", encoding="utf-8") as jf:
                diar = json.load(jf)
            base = f[:-5]  # strip .json
            wav_path = os.path.join(separated_root, call_id, base + ".wav")
            if not os.path.exists(wav_path):
                continue
            audio, sr = sf.read(wav_path)
            segs = {}
            for seg in diar:
                spk = seg["speaker"]
                if spk not in segs:
                    segs[spk] = []
                segs[spk].append(seg)
            seg_meta = []
            txt_lines = []
            # NEW: channel is base (e.g. left-vocals, right-vocals)
            channel = base
            channel_dir = os.path.join(speakers_root, call_id, channel)
            os.makedirs(channel_dir, exist_ok=True)
            for spk, seglist in segs.items():
                spk_dir = os.path.join(channel_dir, spk)
                os.makedirs(spk_dir, exist_ok=True)
                for idx, seg in enumerate(seglist):
                    start = int(seg["start"] * sr)
                    end = int(seg["end"] * sr)
                    seg_audio = audio[start:end]
                    # Ensure mono: Parakeet requires mono input (see https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
                    if seg_audio.ndim == 2:
                        # If stereo, average channels to mono
                        seg_audio = seg_audio.mean(axis=1)
                    # Ensure shape is (samples,)
                    if seg_audio.ndim > 1:
                        seg_audio = np.squeeze(seg_audio)
                    seg_name = f"{idx:04d}-{int(seg['start']*100):07d}-{int(seg['end']*100):07d}.wav"
                    seg_path = os.path.join(spk_dir, seg_name)
                    sf.write(seg_path, seg_audio, sr)
                    meta = {
                        "call_id": call_id,
                        "channel": channel,
                        "speaker": spk,
                        "index": idx,
                        "start": seg["start"],
                        "end": seg["end"],
                        "wav": seg_path,
                    }
                    seg_meta.append(meta)
                    txt_lines.append(f"[{spk}][{seg['start']:.2f}-{seg['end']:.2f}] {seg_path}")
            # Write JSON and TXT logs
            json_log = os.path.join(channel_dir, f"{base}_segments.json")
            txt_log = os.path.join(channel_dir, f"{base}_segments.txt")
            with open(json_log, "w", encoding="utf-8") as jf:
                json.dump(seg_meta, jf, indent=2)
            with open(txt_log, "w", encoding="utf-8") as tf:
                tf.write("\n".join(txt_lines))
            results.extend(seg_meta)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch speaker diarization for separated vocals.")
    parser.add_argument("--separated_root", required=True, help="Root folder of separated/<call id>/")
    parser.add_argument("--diarized_root", required=True, help="Root folder for diarized/<call id>/ outputs")
    parser.add_argument("--hf_token", default=None, help="Hugging Face token (or set via env)")
    parser.add_argument("--min_speakers", type=int, default=None)
    parser.add_argument("--max_speakers", type=int, default=None)
    args = parser.parse_args()
    batch_diarize(
        args.separated_root,
        args.diarized_root,
        hf_token=args.hf_token,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    ) 