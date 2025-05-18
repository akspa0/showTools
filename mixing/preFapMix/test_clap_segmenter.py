import os
from clap_segmenter import CLAPSegmenter

# Minimal test for CLAPSegmenter
# Usage: python test_clap_segmenter.py
# Ensure you have a short test audio file (e.g., test_audio/test.wav) and ffmpeg/ffprobe installed.

def test_clap_segmentation():
    audio_path = "test_audio/test.wav"  # Replace with your test file path
    prompts = ["telephone ringing", "hang-up tone"]
    segmenter = CLAPSegmenter(chunk_duration_s=3)
    metadata = segmenter.segment(audio_path, prompts, confidence_threshold=0.5, output_dir="test_clap_segments")
    print("Segmentation metadata:")
    print(metadata)
    # Basic checks
    assert "segments" in metadata, "No segments in metadata!"
    assert len(metadata["segments"]) > 0, "No segments detected!"
    for seg in metadata["segments"]:
        assert os.path.exists(seg["file"]), f"Segment file missing: {seg['file']}"
    print("CLAP segmentation test passed.")

if __name__ == "__main__":
    test_clap_segmentation() 