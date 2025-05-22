import os
import json
from pathlib import Path
from typing import List, Dict
from transformers import ClapProcessor, ClapModel
import torch
import torchaudio
import re
import numpy as np

def parse_anonymized_filename(filename):
    # Example: 0000-out-20250511-221253.wav
    m = re.match(r'(\d{4})-(out)-([\d-]+)\.wav', filename)
    if not m:
        return None, None, None
    call_id, channel, timestamp = m.groups()
    return call_id, channel, timestamp

def chunk_audio(waveform, sr, chunk_length_sec=5, overlap_sec=2):
    chunk_size = int(sr * chunk_length_sec)
    overlap_size = int(sr * overlap_sec)
    total_samples = waveform.shape[1]
    chunks = []
    start = 0
    while start < total_samples:
        end = min(start + chunk_size, total_samples)
        chunk = waveform[:, start:end]
        if chunk.shape[1] > 0:
            chunks.append((chunk, start / sr, end / sr))  # (waveform, start_time, end_time)
        if end == total_samples:
            break
        start += chunk_size - overlap_size
    return chunks

def run_clap_annotation_on_chunk(chunk_waveform, sr, prompts, model, processor, confidence_threshold=0.5):
    # Convert to mono if not already
    if chunk_waveform.shape[0] > 1:
        chunk_waveform = chunk_waveform.mean(dim=0)
    else:
        chunk_waveform = chunk_waveform.squeeze(0)
    chunk_np = chunk_waveform.cpu().numpy()
    # Pass as a list of 1D numpy arrays
    inputs = processor(text=prompts, audios=[chunk_np], sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_audio = outputs.logits_per_audio[0].softmax(dim=0)
    results = []
    for i, prompt in enumerate(prompts):
        score = logits_per_audio[i].item()
        if score >= confidence_threshold:
            results.append({
                'prompt': prompt,
                'confidence': score
            })
    return results

def run_clap_annotation(audio_path: Path, prompts: List[str], model_id: str = "laion/clap-htsat-unfused", chunk_length_sec=5, overlap_sec=2, confidence_threshold=0.5) -> List[Dict]:
    processor = ClapProcessor.from_pretrained(model_id)
    model = ClapModel.from_pretrained(model_id)
    waveform, sr = torchaudio.load(str(audio_path))
    if sr != 48000:
        waveform = torchaudio.functional.resample(waveform, sr, 48000)
        sr = 48000
    chunks = chunk_audio(waveform, sr, chunk_length_sec, overlap_sec)
    all_results = []
    for idx, (chunk, start_time, end_time) in enumerate(chunks):
        chunk_results = run_clap_annotation_on_chunk(chunk, sr, prompts, model, processor, confidence_threshold)
        for event in chunk_results:
            all_results.append({
                'chunk_index': idx,
                'start_time': start_time,
                'end_time': end_time,
                'prompt': event['prompt'],
                'confidence': event['confidence']
            })
    return all_results

def annotate_clap_for_out_files(renamed_dir: Path, clap_dir: Path, prompts: List[str] = None, model=None, chunk_length_sec=5, overlap_sec=2, confidence_threshold=0.5) -> List[Dict]:
    """
    Annotate all 'out' files in renamed_dir using CLAP and save results in clap_dir.
    Uses the specified CLAP model (default: fused model).
    """
    if model is None:
        model = 'laion/clap-htsat-fused'  # Default to fused model
    # Default prompts if not provided
    if prompts is None:
        prompts = [
            'dog barking', 'DTMF', 'ringing', 'yelling', 'music', 'laughter', 'crying', 'doorbell', 'car horn', 'applause', 'gunshot',
            'siren', 'footsteps', 'phone hangup', 'phone pickup', 'busy signal', 'static', 'noise', 'silence'
        ]
    results = []
    for file in renamed_dir.iterdir():
        if not file.is_file() or '-out-' not in file.name:
            continue
        call_id, channel, timestamp = parse_anonymized_filename(file.name)
        if not call_id:
            continue
        out_dir = clap_dir / call_id
        out_dir.mkdir(parents=True, exist_ok=True)
        annotations = run_clap_annotation(file, prompts, model, chunk_length_sec, overlap_sec, confidence_threshold)
        ann_path = out_dir / 'clap_annotations.json'
        with open(ann_path, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2)
        results.append({
            'call_id': call_id,
            'input_name': file.name,
            'annotation_path': str(ann_path),
            'accepted_annotations': annotations
        })
    return results 