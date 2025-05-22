import os
import json
from pathlib import Path
from typing import List, Dict, Optional
import torch

# For Parakeet
try:
    from nemo.collections.asr.models import ASRModel
except ImportError:
    ASRModel = None
# For Whisper
try:
    import whisper
except ImportError:
    whisper = None

def transcribe_segments(
    segments: List[Dict],
    config: Dict
) -> List[Dict]:
    """
    Transcribe a list of speaker segments using Parakeet or Whisper.
    Each segment dict must have: call_id, channel, speaker, index, start, end, wav
    Config must specify asr_engine ('parakeet' or 'whisper'), model name, device, etc.
    Returns a list of transcript metadata dicts.
    """
    asr_engine = config.get('asr_engine', 'whisper')
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    language = config.get('language', None)
    results = []

    if asr_engine == 'parakeet':
        if ASRModel is None:
            raise ImportError("nemo_toolkit is not installed. Cannot use Parakeet ASR.")
        model_name = config.get('parakeet_model', 'nvidia/parakeet-tdt-0.6b-v2')
        asr_model = ASRModel.from_pretrained(model_name=model_name, map_location=device)
        asr_model.eval()
        for seg in segments:
            wav_path = seg['wav']
            out_base = Path(wav_path).with_suffix("")
            try:
                # Disable timestamps: segments are already timestamped by the pipeline
                output = asr_model.transcribe([wav_path], batch_size=1, return_hypotheses=True, timestamps=False)[0]
                text = getattr(output, 'text', None)
                words = getattr(output, 'words', None)
                # Save TXT
                txt_path = str(out_base) + '.txt'
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"{seg['speaker']} [{seg['start']:.2f} --> {seg['end']:.2f}]: {text}\n")
                # Save JSON
                json_path = str(out_base) + '.json'
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        **seg,
                        'asr_engine': 'parakeet',
                        'text': text,
                        'words': words
                    }, f, indent=2, ensure_ascii=False)
                results.append({
                    **seg,
                    'asr_engine': 'parakeet',
                    'text': text,
                    'words': words,
                    'txt': txt_path,
                    'json': json_path
                })
            except Exception as e:
                results.append({**seg, 'asr_engine': 'parakeet', 'error': str(e)})
    elif asr_engine == 'whisper':
        if whisper is None:
            raise ImportError("openai-whisper is not installed.")
        model_name = config.get('whisper_model', 'base.en')
        whisper_model = whisper.load_model(model_name, device=device)
        for seg in segments:
            wav_path = seg['wav']
            out_base = Path(wav_path).with_suffix("")
            try:
                result = whisper_model.transcribe(wav_path, language=language, verbose=False)
                text = result.get('text', '').strip()
                # Save TXT
                txt_path = str(out_base) + '.txt'
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"{seg['speaker']} [{seg['start']:.2f} --> {seg['end']:.2f}]: {text}\n")
                # Save JSON
                json_path = str(out_base) + '.json'
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        **seg,
                        'asr_engine': 'whisper',
                        'text': text
                    }, f, indent=2, ensure_ascii=False)
                results.append({
                    **seg,
                    'asr_engine': 'whisper',
                    'text': text,
                    'txt': txt_path,
                    'json': json_path
                })
            except Exception as e:
                results.append({**seg, 'asr_engine': 'whisper', 'error': str(e)})
    else:
        raise ValueError(f"Unknown asr_engine: {asr_engine}")
    return results 