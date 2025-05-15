import argparse
from pathlib import Path
from .pipeline import UnifiedPipeline
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / 'ClapAnnotator'))
from ClapAnnotator.config.settings import AUDIO_SEPARATOR_AVAILABLE_MODELS, DEFAULT_AUDIO_SEPARATOR_MODEL

def main():
    parser = argparse.ArgumentParser(description="Unified mhrpTools Pipeline CLI")
    # Separator model choices
    separator_model_choices = list(AUDIO_SEPARATOR_AVAILABLE_MODELS.keys())
    parser.add_argument("--input", required=True, help="Input audio file or folder")
    parser.add_argument("--show-name", required=True, help="Name for the show")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs")
    parser.add_argument("--separator-model", choices=separator_model_choices, default=DEFAULT_AUDIO_SEPARATOR_MODEL, help="Vocal separator model to use")
    # CLAP options
    parser.add_argument("--clap-model", default=None, help="CLAP model name (HuggingFace)")
    parser.add_argument("--clap-chunk-duration", type=int, default=None, help="CLAP chunk duration (seconds)")
    parser.add_argument("--clap-confidence", type=float, default=None, help="CLAP confidence threshold")
    parser.add_argument("--clap-prompts", nargs='+', default=None, help="CLAP prompts (space-separated list)")
    # Mixing options
    parser.add_argument("--mix-normalize", action='store_true', help="Enable loudness normalization in mixing")
    parser.add_argument("--mix-no-normalize", action='store_true', help="Disable loudness normalization in mixing")
    parser.add_argument("--mix-tones", action='store_true', help="Insert tones.wav between calls in show-edit mode")
    parser.add_argument("--mix-lufs", type=float, default=None, help="Target LUFS for normalization")
    # Transcription options
    parser.add_argument("--wb-model", default=None, help="WhisperBite model name")
    parser.add_argument("--wb-num-speakers", type=int, default=None, help="Number of speakers for diarization")
    parser.add_argument("--wb-enable-word-extraction", action='store_true', help="Enable word-level extraction in WhisperBite")
    parser.add_argument("--wb-enable-second-pass", action='store_true', help="Enable second-pass diarization in WhisperBite")
    parser.add_argument("--wb-auto-speakers", action='store_true', help="Enable automatic speaker count detection in WhisperBite")
    parser.add_argument("--wb-attempt-sound-detection", action='store_true', help="Enable sound event detection in WhisperBite")
    args = parser.parse_args()

    # Handle normalization flag logic
    mix_normalize = True
    if args.mix_no_normalize:
        mix_normalize = False
    elif args.mix_normalize:
        mix_normalize = True

    pipeline = UnifiedPipeline(
        show_name=args.show_name,
        output_dir=Path(args.output_dir),
        separator_model=args.separator_model,
        clap_model=args.clap_model if args.clap_model else None,
        clap_chunk_duration=args.clap_chunk_duration if args.clap_chunk_duration else None,
        clap_confidence=args.clap_confidence if args.clap_confidence else None,
        clap_prompts=args.clap_prompts if args.clap_prompts else None,
        mix_normalize=mix_normalize,
        mix_tones=args.mix_tones,
        mix_lufs=args.mix_lufs if args.mix_lufs else None,
        wb_model=args.wb_model if args.wb_model else None,
        wb_num_speakers=args.wb_num_speakers if args.wb_num_speakers else None,
        wb_enable_word_extraction=args.wb_enable_word_extraction,
        wb_enable_second_pass=args.wb_enable_second_pass,
        wb_auto_speakers=args.wb_auto_speakers,
        wb_attempt_sound_detection=args.wb_attempt_sound_detection
    )
    input_path = Path(args.input)
    if input_path.is_file():
        pipeline.process_audio_file(input_path)
    elif input_path.is_dir():
        for audio_file in input_path.rglob("*.wav"):
            pipeline.process_audio_file(audio_file)
    else:
        print(f"Input path {input_path} does not exist.")
        return
    pipeline.finalize_show()

if __name__ == "__main__":
    main() 