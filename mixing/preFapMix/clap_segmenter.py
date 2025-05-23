import subprocess
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from transformers import AutoProcessor, AutoModel

class CLAPSegmenter:
    """
    CLAP-based audio segmentation using Hugging Face transformers CLAP model.
    All audio I/O is performed via ffmpeg/ffprobe subprocess calls.
    """
    def __init__(self, model_name: str = "laion/clap-htsat-fused", device: Optional[str] = None, chunk_duration_s: int = 3):
        self.model_name = model_name
        self.chunk_duration_s = chunk_duration_s
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

    @staticmethod
    def get_audio_duration(audio_path: str) -> float:
        """Get duration of audio file using ffprobe."""
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", audio_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout.strip()
        if result.returncode != 0 or not output or output == 'N/A':
            raise RuntimeError(f"ffprobe error or invalid duration for file '{audio_path}':\n  Output: '{output}'\n  Stderr: '{result.stderr.strip()}'\n  Command: {' '.join(cmd)}")
        try:
            return float(output)
        except ValueError:
            raise RuntimeError(f"ffprobe returned non-numeric duration for file '{audio_path}':\n  Output: '{output}'\n  Stderr: '{result.stderr.strip()}'\n  Command: {' '.join(cmd)}")

    @staticmethod
    def extract_audio_chunk(input_path: str, output_path: str, start: float, duration: float):
        """Extract a chunk from audio using ffmpeg."""
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(start), "-t", str(duration),
            "-i", input_path, "-ac", "1", "-ar", "48000", output_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {result.stderr.decode()}")

    @staticmethod
    def wav_to_numpy(wav_path: str, sr: int = 48000) -> np.ndarray:
        """
        Read a WAV file into a numpy array using ffmpeg (no soundfile/librosa).
        Returns a float32 mono waveform at the given sample rate.
        """
        cmd = [
            "ffmpeg", "-v", "error", "-i", wav_path,
            "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-"
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg error reading wav to numpy: {proc.stderr.decode()}")
        audio = np.frombuffer(proc.stdout, dtype=np.float32)
        return audio

    def segment(self, audio_path: str, prompts: List[str], confidence_threshold: float = 0.5, output_dir: Optional[str] = None) -> Dict:
        """
        Segment audio using CLAP event detection and prompt pairing.
        Args:
            audio_path: Path to input audio file.
            prompts: List of event prompts (e.g., ["telephone ringing", "hang-up tone"])
            confidence_threshold: Minimum probability to consider a detection.
            output_dir: Directory to save segments and metadata.
        Returns:
            Dictionary with segment metadata and detection results.
        """
        audio_path = str(audio_path)
        duration = self.get_audio_duration(audio_path)
        chunk_len = self.chunk_duration_s
        n_chunks = int(np.ceil(duration / chunk_len))
        output_dir = Path(output_dir or f"{Path(audio_path).stem}_clap_segments")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Precompute text features
        text_inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
        text_features = self.model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        detections = []
        for i in range(n_chunks):
            start = i * chunk_len
            end = min((i + 1) * chunk_len, duration)
            chunk_path = output_dir / f"chunk_{i:03d}.wav"
            self.extract_audio_chunk(audio_path, str(chunk_path), start, end - start)
            # Read chunk audio as numpy array using ffmpeg
            chunk_audio = self.wav_to_numpy(str(chunk_path), sr=48000)
            audio_inputs = self.processor(audios=chunk_audio, sampling_rate=48000, return_tensors="pt").to(self.device)
            audio_features = self.model.get_audio_features(**audio_inputs)
            audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
            logits_per_audio = audio_features @ text_features.T
            probs = torch.sigmoid(logits_per_audio).squeeze().cpu().detach().numpy()
            if probs.ndim == 0:
                probs = np.array([probs])
            chunk_probabilities = {prompt: float(prob) for prompt, prob in zip(prompts, probs)}
            filtered_prompts = {p: prob for p, prob in chunk_probabilities.items() if prob >= confidence_threshold}
            if filtered_prompts:
                detections.append({
                    "start_time_s": round(start, 3),
                    "end_time_s": round(end, 3),
                    "prompts": filtered_prompts
                })
            chunk_path.unlink()  # Remove chunk after processing

        # Pair events and create segments
        segments = self.pair_events(detections, prompts, duration)
        # Extract segments using ffmpeg
        segment_files = []
        for idx, seg in enumerate(segments):
            seg_path = output_dir / f"segment_{idx+1:03d}_{seg['label']}_{int(seg['start'])}s_{int(seg['end'])}s.wav"
            self.extract_audio_chunk(audio_path, str(seg_path), seg['start'], seg['end'] - seg['start'])
            seg['file'] = str(seg_path)
            segment_files.append(seg)

        # Save metadata
        metadata = {
            "input_file": audio_path,
            "prompts": prompts,
            "confidence_threshold": confidence_threshold,
            "detections": detections,
            "segments": segment_files
        }
        with open(output_dir / "clap_segments_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        return metadata

    @staticmethod
    def pair_events(detections: List[Dict], prompts: List[str], audio_duration: float) -> List[Dict]:
        """
        Pair detected events to form segments. Handles bookends and unmatched regions.
        For example, if prompts = ["start", "end"], segments are from each start to next end.
        """
        # Example: pair first prompt as start, second as end
        if len(prompts) < 2:
            # If only one prompt, treat each detection as a segment
            return [
                {"start": d["start_time_s"], "end": d["end_time_s"], "label": list(d["prompts"].keys())[0]}
                for d in detections
            ]
        start_prompt, end_prompt = prompts[0], prompts[1]
        starts = [d for d in detections if start_prompt in d["prompts"]]
        ends = [d for d in detections if end_prompt in d["prompts"]]
        segments = []
        i, j = 0, 0
        # Bookend: before first start
        if starts and starts[0]["start_time_s"] > 0:
            segments.append({"start": 0, "end": starts[0]["start_time_s"], "label": "bookend_start"})
        while i < len(starts) and j < len(ends):
            s = starts[i]["start_time_s"]
            # Find the next end after this start
            while j < len(ends) and ends[j]["start_time_s"] <= s:
                j += 1
            if j < len(ends):
                e = ends[j]["end_time_s"]
                segments.append({"start": s, "end": e, "label": f"{start_prompt}_to_{end_prompt}"})
                i += 1
                j += 1
            else:
                # No more ends, bookend to end of audio
                segments.append({"start": s, "end": audio_duration, "label": "bookend_end"})
                i += 1
        # Bookend: after last end
        if ends and ends[-1]["end_time_s"] < audio_duration:
            segments.append({"start": ends[-1]["end_time_s"], "end": audio_duration, "label": "bookend_end"})
        return segments

# --- Pipeline-compatible wrapper function ---
def segment(audio_path, prompts, confidence_threshold=0.5, output_dir=None, **kwargs):
    """
    Pipeline-compatible wrapper for CLAP segmentation.
    Instantiates CLAPSegmenter and calls its segment method.
    Args:
        audio_path: Path to input audio file.
        prompts: List of event prompts.
        confidence_threshold: Minimum probability to consider a detection.
        output_dir: Directory to save segments and metadata.
        **kwargs: Ignored extra arguments for compatibility.
    Returns:
        Dict with 'clap_events_file' (path to metadata JSON) and full metadata.
    """
    segmenter = CLAPSegmenter()
    metadata = segmenter.segment(audio_path, prompts, confidence_threshold=confidence_threshold, output_dir=output_dir)
    # Path to the main metadata file
    output_dir_path = Path(output_dir or f"{Path(audio_path).stem}_clap_segments")
    metadata_path = output_dir_path / "clap_segments_metadata.json"
    return {"clap_events_file": str(metadata_path), "clap_segments_metadata": metadata}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CLAP-based audio segmentation")
    parser.add_argument("--audio", required=True, help="Path to input audio file")
    parser.add_argument("--prompts", nargs='+', required=True, help="List of prompts for event detection")
    parser.add_argument("--outdir", default=None, help="Output directory for segments and metadata")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for detection")
    parser.add_argument("--chunk", type=int, default=3, help="Chunk duration in seconds")
    args = parser.parse_args()

    segmenter = CLAPSegmenter(chunk_duration_s=args.chunk)
    metadata = segmenter.segment(args.audio, args.prompts, confidence_threshold=args.threshold, output_dir=args.outdir)
    print(json.dumps(metadata, indent=2)) 