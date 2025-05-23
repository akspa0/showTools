import os
import json
from pathlib import Path
from typing import List, Dict
from transformers import ClapProcessor, ClapModel
import torch
import torchaudio
import re
import numpy as np
import soundfile as sf
from shutil import copyfile
import logging

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
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logging.info(f"[CLAP] Model loaded: {model_id} on device: {device}")
    model.eval()
    try:
        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.numel() == 0:
            raise ValueError("Audio file is empty (0 elements)")
        if sr != 48000:
            waveform = torchaudio.functional.resample(waveform, sr, 48000)
            sr = 48000
    except Exception as e:
        # Raise a controlled error to be caught by the caller
        raise RuntimeError(f"Failed to load or process audio: {audio_path.name} ({str(e)})")
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

def annotate_clap_for_out_files(
    input_dir: Path,
    output_dir: Path,
    prompts: list = None,
    model_id: str = "laion/clap-htsat-unfused",
    chunk_length_sec=5,
    overlap_sec=2,
    confidence_threshold=0.6
) -> list:
    """
    Annotate all files in input_dir using CLAP and save results in output_dir.
    Uses the specified CLAP model.
    Logs all CLAP events with confidence >= threshold to a .json per file.
    Returns a list of result dicts for manifest.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    # Load model and processor
    logging.info(f"[CLAP] Loading model: {model_id}")
    processor = ClapProcessor.from_pretrained(model_id)
    model = ClapModel.from_pretrained(model_id)
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logging.info(f"[CLAP] Model loaded: {model_id} on device: {device}")
    model.eval()
    for audio_file in input_dir.glob('*'):
        if not audio_file.is_file() or audio_file.suffix.lower() not in ['.wav', '.mp3', '.flac']:
            continue
        # Load audio to check sample rate
        waveform, sr = torchaudio.load(str(audio_file))
        duration = waveform.shape[-1] / sr
        logging.info(f"[CLAP] Analyzing: {audio_file} | shape: {waveform.shape} | sr: {sr} | duration: {duration:.2f}s")
        # Resample to 48000 Hz using ffmpeg if needed
        target_sr = 48000
        resampled_audio_path = output_dir / (audio_file.stem + '_clap_48k.wav')
        if sr != target_sr:
            import subprocess
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-i', str(audio_file),
                '-ar', str(target_sr),
                '-ac', '1',  # mono for CLAP
                str(resampled_audio_path)
            ]
            logging.info(f"[CLAP] Running ffmpeg for resample: {' '.join(ffmpeg_cmd)}")
            result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                logging.error(f"[CLAP] ffmpeg resample failed for {audio_file.name}: {result.stderr.decode()}")
                continue
            logging.info(f"[CLAP] Saved ffmpeg-resampled audio to {resampled_audio_path}")
        else:
            # Save a copy as the resampled path for consistency
            import shutil
            shutil.copy2(audio_file, resampled_audio_path)
            logging.info(f"[CLAP] Audio already at 48kHz, copied to {resampled_audio_path}")
        # Load the resampled audio
        waveform, sr = torchaudio.load(str(resampled_audio_path))
        # Convert to mono if needed (should already be mono from ffmpeg)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Chunking (memory-efficient)
        chunk_samples = int(chunk_length_sec * sr)
        overlap_samples = int(overlap_sec * sr)
        total_samples = waveform.shape[1]
        start = 0
        chunk_idx = 0
        events = []
        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            chunk = waveform[:, start:end]
            chunk_duration = (end - start) / sr
            if chunk.shape[1] < chunk_samples // 2:
                start += chunk_samples - overlap_samples
                continue
            logging.info(f"[CLAP] Processing chunk {chunk_idx}: {start/sr:.2f}s - {end/sr:.2f}s ({chunk_duration:.2f}s)")
            # Prepare input for CLAP
            audio_np = chunk.squeeze().numpy()
            try:
                inputs = processor(audios=audio_np, return_tensors="pt", sampling_rate=sr, padding=True)
                # Move all tensors to the same device as the model
                inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                for prompt in prompts or []:
                    with torch.no_grad():
                        text_inputs = processor(text=prompt, return_tensors="pt", padding=True)
                        text_inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in text_inputs.items()}
                        outputs = model(**inputs, **text_inputs)
                        logits = outputs.logits_per_audio
                        conf = logits.item() if hasattr(logits, 'item') else float(logits)
                        if conf >= confidence_threshold:
                            event = {
                                'file': str(audio_file),
                                'prompt': prompt,
                                'confidence': conf,
                                'start_time': start / sr,
                                'end_time': end / sr
                            }
                            events.append(event)
            except Exception as e:
                logging.error(f"[CLAP] Error processing chunk {chunk_idx}: {e}")
            start += chunk_samples - overlap_samples
            chunk_idx += 1
        logging.info(f"[CLAP] Processed {chunk_idx} chunks for {audio_file.name}")
        # Write results
        out_json = output_dir / (audio_file.stem + '_clap_annotations.json')
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(events, f, indent=2)
        logging.info(f"[CLAP] {audio_file.name}: {len(events)} events detected, written to {out_json}")
        results.append({
            'input_name': audio_file.name,
            'annotation_path': str(out_json),
            'resampled_audio_path': str(resampled_audio_path),
            'accepted_annotations': events,
            'call_id': audio_file.stem
        })
    return results

def pair_clap_events(events, config, total_duration):
    """
    Pair start and end events using config parameters for robust segmentation.
    Returns a list of (start, end) tuples.
    """
    start_prompt = config.get("start_prompt", "telephone ring tones")
    end_prompt = config.get("end_prompt", "hang-up tones")
    min_gap = config.get("min_gap_sec", 2.0)
    min_call = config.get("min_call_duration_sec", 10.0)
    noise_gap = config.get("noise_gap_sec", 1.5)
    pairs = []
    i = 0
    n = len(events)
    while i < n:
        e = events[i]
        if e["prompt"] == start_prompt:
            start_time = e["start_time"]
            # Find the next end_prompt after this start
            j = i + 1
            while j < n:
                e2 = events[j]
                if e2["prompt"] == end_prompt:
                    end_time = e2["start_time"]
                    # Ignore if too close (noise)
                    if end_time - start_time < noise_gap:
                        j += 1
                        continue
                    # Only accept if duration is long enough
                    if end_time - start_time >= min_call:
                        pairs.append((start_time, end_time))
                        i = j  # Move to after this end
                        break
                j += 1
            else:
                # No end found, segment goes to file end
                if total_duration - start_time >= min_call:
                    pairs.append((start_time, total_duration))
                i = n  # Done
                break
        i += 1
    return pairs

def segment_audio_with_clap(
    audio_path: Path,
    segmentation_config: dict,
    output_dir: Path,
    model_id: str = "laion/clap-htsat-unfused",
    chunk_length_sec=5,
    overlap_sec=2
) -> list:
    """
    Segment a long audio file into calls using CLAP-based event detection and intelligent pairing.
    Returns a list of segment metadata dicts.
    """
    prompts = segmentation_config.get("prompts", ["telephone ring tones", "hang-up tones"])
    confidence_threshold = segmentation_config.get("confidence_threshold", 0.6)
    min_segment_length = segmentation_config.get("min_segment_length_sec", 10)
    padding = segmentation_config.get("segment_padding_sec", 0.5)

    # Run CLAP event detection
    events = run_clap_annotation(
        audio_path,
        prompts,
        model_id=model_id,
        chunk_length_sec=chunk_length_sec,
        overlap_sec=overlap_sec,
        confidence_threshold=confidence_threshold
    )
    # Only keep events with relevant prompts
    start_prompt = segmentation_config.get("start_prompt", "telephone ring tones")
    end_prompt = segmentation_config.get("end_prompt", "hang-up tones")
    filtered_events = [e for e in events if e["prompt"] in (start_prompt, end_prompt)]
    # Load audio
    waveform, sr = torchaudio.load(str(audio_path))
    total_duration = waveform.shape[1] / sr
    # Pair events intelligently
    pairs = pair_clap_events(filtered_events, segmentation_config, total_duration)
    # Build segments
    segments = []
    for i, (seg_start, seg_end) in enumerate(pairs):
        seg_start = max(0.0, seg_start - padding)
        seg_end = min(total_duration, seg_end + padding)
        if seg_end - seg_start < min_segment_length:
            continue
        seg_wave = waveform[:, int(seg_start * sr):int(seg_end * sr)]
        seg_name = f"{audio_path.stem}-seg{i:04d}.wav"
        seg_path = output_dir / seg_name
        torchaudio.save(str(seg_path), seg_wave, sr)
        segments.append({
            "segment_index": i,
            "start": seg_start,
            "end": seg_end,
            "output_path": str(seg_path),
            "source_file": str(audio_path),
            "events": [e for e in filtered_events if seg_start <= e["start_time"] < seg_end]
        })
    return segments 