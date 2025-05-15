import logging
import ffmpeg # Use ffmpeg-python library
from pathlib import Path

log = logging.getLogger(__name__)

def resample_audio_ffmpeg(input_path: Path, output_path: Path, target_sr: int):
    """Resamples an audio file to the target sample rate using ffmpeg,
    converting to mono and saving as 16-bit PCM WAV.

    Args:
        input_path: Path to the input audio file.
        output_path: Path to save the resampled audio file.
        target_sr: The target sample rate in Hz.

    Returns:
        Path to the output file if successful.

    Raises:
        FileNotFoundError: If the input file does not exist.
        RuntimeError: If ffmpeg command fails.
    """
    if not input_path.is_file():
        log.error(f"Input file for resampling not found: {input_path}")
        raise FileNotFoundError(f"Input file not found: {input_path}")

    log.info(f"Resampling '{input_path}' to {target_sr}Hz -> '{output_path}'")
    try:
        (
            ffmpeg
            .input(str(input_path)) # ffmpeg-python prefers string paths
            .output(
                str(output_path), 
                ar=str(target_sr),  # Set audio sample rate
                ac=1,               # Set number of audio channels to 1 (mono)
                acodec='pcm_s16le'  # Set audio codec to PCM signed 16-bit little-endian (standard WAV)
            )
            .overwrite_output() # Overwrite output file if it exists
            .run(capture_stdout=True, capture_stderr=True, quiet=True) # Use quiet=True to suppress console output unless error
        )
        log.info(f"Successfully resampled to {output_path}")
        return output_path
    except ffmpeg.Error as e:
        stderr_output = e.stderr.decode('utf8').strip() if e.stderr else "No stderr output"
        log.error(f"ffmpeg resampling failed for {input_path}. Error: {stderr_output}")
        # Clean up potentially corrupted output file
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                log.warning(f"Could not delete potentially corrupt output file: {output_path}")
        raise RuntimeError(f"ffmpeg resampling failed: {stderr_output}") from e 