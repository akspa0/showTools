import subprocess
from pathlib import Path
import logging

class AudioSeparator:
    """Standalone audio separator using Demucs (no legacy imports)."""
    def __init__(self, model_name="htdemucs", output_dir=None):
        self.model_name = model_name
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def separate(self, input_path, output_dir=None):
        input_path = Path(input_path)
        out_dir = Path(output_dir) if output_dir else self.output_dir
        if not out_dir:
            raise ValueError("Output directory must be specified.")
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            cmd = [
                "demucs",
                "--two-stems", "vocals",
                "-n", self.model_name,
                "-o", str(out_dir),
                str(input_path)
            ]
            subprocess.run(cmd, check=True)
            # Demucs output is in out_dir/model_name/input_basename/vocals.wav, no_vocals.wav
            demucs_out = out_dir / self.model_name / input_path.stem
            vocals = demucs_out / "vocals.wav"
            no_vocals = demucs_out / "no_vocals.wav"
            result = {}
            if vocals.exists():
                result["Vocals"] = vocals
            if no_vocals.exists():
                result["NoVocals"] = no_vocals
            return result
        except Exception as e:
            logging.error(f"[Separation Error] {input_path}: {e}")
            return {} 