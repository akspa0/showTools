from pathlib import Path
from typing import Optional, List
from .data_model import AudioFile, Call, Show

# --- Separation & Annotation Stage Imports ---
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / 'ClapAnnotator'))
from ClapAnnotator.audio_separation.separator import AudioSeparatorWrapper
from ClapAnnotator.config import settings as clap_settings
from ClapAnnotator.utils.file_utils import ensure_dir
from ClapAnnotator.clap_annotation.annotator import CLAPAnnotatorWrapper

# --- Mixing Stage Imports ---
sys.path.append(str(Path(__file__).resolve().parent.parent / 'preFapMix'))
from preFapMix import preFapMix

# --- Transcription Stage Imports ---
sys.path.append(str(Path(__file__).resolve().parent.parent / 'WhisperBite'))
from WhisperBite.whisperBite import process_audio as whisperbite_process_audio

import subprocess
import yaml
import datetime
import tempfile
import shutil
import requests

DEFAULT_SEPARATOR_MODEL = "Mel Band RoFormer Vocals"
DEFAULT_CLAP_MODEL = clap_settings.CLAP_MODEL_NAME
DEFAULT_CLAP_CHUNK_DURATION = clap_settings.CLAP_CHUNK_DURATION_S
DEFAULT_CLAP_CONFIDENCE = clap_settings.DEFAULT_CLAP_CONFIDENCE_THRESHOLD
DEFAULT_CLAP_PROMPTS = ["telephone noises", "dial tone", "ring tone", "telephone interference"]
DEFAULT_MIX_NORMALIZE = True
DEFAULT_MIX_TONES = False
DEFAULT_MIX_LUFS = -14.0
DEFAULT_WB_MODEL = "large-v3"
DEFAULT_WB_NUM_SPEAKERS = 2

class UnifiedPipeline:
    """Main orchestrator for the unified mhrpTools pipeline."""
    def __init__(self, show_name: str, output_dir: Path,
                 separator_model: str = DEFAULT_SEPARATOR_MODEL,
                 clap_model: str = DEFAULT_CLAP_MODEL,
                 clap_chunk_duration: int = DEFAULT_CLAP_CHUNK_DURATION,
                 clap_confidence: float = DEFAULT_CLAP_CONFIDENCE,
                 clap_prompts: Optional[List[str]] = None,
                 mix_normalize: bool = DEFAULT_MIX_NORMALIZE,
                 mix_tones: bool = DEFAULT_MIX_TONES,
                 mix_lufs: float = DEFAULT_MIX_LUFS,
                 wb_model: str = DEFAULT_WB_MODEL,
                 wb_num_speakers: int = DEFAULT_WB_NUM_SPEAKERS,
                 wb_enable_word_extraction: bool = False,
                 wb_enable_second_pass: bool = False,
                 wb_auto_speakers: bool = False,
                 wb_attempt_sound_detection: bool = False):
        self.show = Show(show_name)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.separator_model = separator_model
        self.clap_model = clap_model
        self.clap_chunk_duration = clap_chunk_duration
        self.clap_confidence = clap_confidence
        self.clap_prompts = clap_prompts or DEFAULT_CLAP_PROMPTS
        self.mix_normalize = mix_normalize
        self.mix_tones = mix_tones
        self.mix_lufs = mix_lufs
        self.wb_model = wb_model
        self.wb_num_speakers = wb_num_speakers
        self.wb_enable_word_extraction = wb_enable_word_extraction
        self.wb_enable_second_pass = wb_enable_second_pass
        self.wb_auto_speakers = wb_auto_speakers
        self.wb_attempt_sound_detection = wb_attempt_sound_detection
        # Prepare a directory for separated outputs
        self.separation_dir = self.output_dir / "separated"
        ensure_dir(self.separation_dir)
        # Prepare a directory for mixed outputs
        self.mixed_dir = self.output_dir / "mixed"
        ensure_dir(self.mixed_dir)
        # Prepare a directory for transcription outputs
        self.transcript_dir = self.output_dir / "transcripts"
        ensure_dir(self.transcript_dir)
        # Instantiate the separator wrapper once for efficiency
        self.separator = AudioSeparatorWrapper(
            model_name=self.separator_model,
            output_dir=self.separation_dir,
            model_file_dir=clap_settings.AUDIO_SEPARATOR_MODEL_DIR,
            log_level=clap_settings.LOG_LEVEL
        )
        # Instantiate the CLAP annotator wrapper
        self.clap_annotator = CLAPAnnotatorWrapper(
            model_name=self.clap_model,
            chunk_duration_s=self.clap_chunk_duration,
            expected_sr=clap_settings.CLAP_EXPECTED_SR
        )

    def _download_and_extract_audio(self, url: str, temp_dir: Path) -> Optional[Path]:
        """Download audio/video from URL and extract audio as WAV."""
        try:
            # Download with yt-dlp
            out_path = temp_dir / "downloaded"
            out_path.mkdir(exist_ok=True)
            audio_file = out_path / "input"
            cmd = [
                "yt-dlp", "-f", "bestaudio/best",
                "--extract-audio", "--audio-format", "wav",
                "-o", str(audio_file) + ".%(ext)s",
                url
            ]
            subprocess.run(cmd, check=True)
            # Find the downloaded file
            for f in out_path.glob("input.*"):
                if f.suffix in [".wav", ".mp3", ".m4a", ".aac", ".ogg"]:
                    if f.suffix != ".wav":
                        # Convert to WAV
                        wav_path = out_path / "input.wav"
                        subprocess.run(["ffmpeg", "-y", "-i", str(f), str(wav_path)], check=True)
                        return wav_path
                    return f
            return None
        except Exception as e:
            print(f"[URL Download Error] {url}: {e}")
            return None

    def _extract_audio_from_video(self, video_path: Path, temp_dir: Path) -> Optional[Path]:
        """Extract audio from video file as WAV."""
        try:
            wav_path = temp_dir / (video_path.stem + ".wav")
            cmd = ["ffmpeg", "-y", "-i", str(video_path), str(wav_path)]
            subprocess.run(cmd, check=True)
            return wav_path
        except Exception as e:
            print(f"[Video Extraction Error] {video_path}: {e}")
            return None

    def process_input(self, input_path_or_url: str) -> List[Path]:
        """Handle file, folder, video, or URL input. Returns list of audio file paths to process."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            p = Path(input_path_or_url)
            if p.exists():
                if p.is_file():
                    if p.suffix.lower() in [".mp4", ".mkv", ".mov", ".avi", ".webm"]:
                        # Video file: extract audio
                        audio_path = self._extract_audio_from_video(p, temp_dir)
                        return [audio_path] if audio_path else []
                    else:
                        return [p]
                elif p.is_dir():
                    # Folder: process all audio files
                    return list(p.rglob("*.wav")) + list(p.rglob("*.mp3"))
            elif input_path_or_url.startswith("http://") or input_path_or_url.startswith("https://"):
                # URL: download and extract audio
                audio_path = self._download_and_extract_audio(input_path_or_url, temp_dir)
                return [audio_path] if audio_path else []
            else:
                print(f"[Input Error] Path or URL not found: {input_path_or_url}")
                return []
        except Exception as e:
            print(f"[Input Handling Error] {input_path_or_url}: {e}")
            return []

    def process_audio_file(self, input_path_or_url: str):
        """Process input (file, folder, video, or URL) through all pipeline stages."""
        audio_files = self.process_input(input_path_or_url)
        for audio_path in audio_files:
            if audio_path and audio_path.exists():
                self.process_audio_file(audio_path)

    def finalize_show(self):
        """Finalize the show: concatenate calls, insert tones.wav between calls, and output metadata."""
        try:
            # 1. Collect all mixed stereo call files in order
            call_files = []
            call_metadata = []
            for call in self.show.calls:
                # Use the first stereo file for each call
                stereo_files = [f for k, f in call.mixed_outputs.items() if 'stereo' in k]
                if stereo_files:
                    call_files.append(stereo_files[0])
                    call_metadata.append({
                        'input': str(call.input_audio.path),
                        'stereo_file': str(stereo_files[0]),
                        'clap_annotations': call.clap_annotations,
                        'transcripts': call.transcripts,
                        'separated_stems': {k: str(v) for k, v in call.separated_stems.items()},
                    })
            if not call_files:
                print("[Show-Edit] No mixed stereo call files found. Cannot build show.")
                return
            # 2. Sort files chronologically by input filename or timestamp
            call_files = sorted(call_files, key=lambda f: f.stat().st_mtime)
            # 3. Build concat list: call1, tones, call2, tones, ..., callN (no tones after last call)
            tones_path = Path(__file__).resolve().parent.parent / 'preFapMix' / 'tones.wav'
            concat_list_path = self.output_dir / 'show_concat_list.txt'
            with open(concat_list_path, 'w') as f:
                for i, call_file in enumerate(call_files):
                    f.write(f"file '{call_file}'\n")
                    if i < len(call_files) - 1:
                        f.write(f"file '{tones_path}'\n")
            # 4. Output show file
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
            show_file = self.output_dir / f"{self.show.name}_{timestamp}.wav"
            cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', str(concat_list_path),
                '-c', 'copy',
                str(show_file)
            ]
            subprocess.run(cmd, check=True)
            print(f"[Show-Edit] Show file created: {show_file}")
            # 5. Output metadata as YAML
            show_metadata = {
                'show': {
                    'name': self.show.name,
                    'file': str(show_file),
                    'calls': call_metadata,
                    'tones_file': str(tones_path),
                    'concat_list': [str(f) for f in call_files],
                    'created': timestamp
                }
            }
            metadata_path = self.output_dir / f"{self.show.name}_{timestamp}_metadata.yaml"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                yaml.dump(show_metadata, f, allow_unicode=True, sort_keys=False)
            print(f"[Show-Edit] Metadata written: {metadata_path}")
        except Exception as e:
            print(f"[Show-Edit Error] {e}") 