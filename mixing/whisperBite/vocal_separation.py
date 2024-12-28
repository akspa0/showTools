import os
import logging
import subprocess

def separate_vocals_with_demucs(input_audio, output_dir):
    """Separate vocals from an audio file using Demucs."""
    demucs_output_dir = os.path.join(output_dir, "demucs")
    os.makedirs(demucs_output_dir, exist_ok=True)

    try:
        logging.info(f"Running Demucs on {input_audio}. Output directory: {demucs_output_dir}")
        subprocess.run(
            [
                "demucs",
                "--two-stems", "vocals",
                "--out", demucs_output_dir,
                input_audio
            ],
            check=True
        )
        model_dir = os.path.join(demucs_output_dir, "htdemucs")
        input_base_name = os.path.splitext(os.path.basename(input_audio))[0]
        vocals_dir = os.path.join(model_dir, input_base_name)
        vocals_file = os.path.join(vocals_dir, "vocals.wav")

        if not os.path.exists(vocals_file):
            logging.error(f"Demucs output file not found: {vocals_file}")
            raise FileNotFoundError(f"Expected vocals file not found at {vocals_file}")

        logging.info(f"Vocal separation completed successfully. File saved to {vocals_file}")
        return vocals_file
    except Exception as e:
        logging.error(f"Error during vocal separation: {e}")
        raise
