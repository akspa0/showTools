import argparse
import os
from mhrpTools.core import mhrp_pipeline

def main():
    parser = argparse.ArgumentParser(description="mhrpTools: Unified Audio Processing CLI")
    parser.add_argument("input", help="Input audio file or folder")
    parser.add_argument("--output-dir", default="mhrpTools_Output", help="Directory to save outputs")
    parser.add_argument("--mode", choices=["auto", "mixing", "soundbites", "both"], default="auto", help="Processing mode")
    parser.add_argument("--process-left-channel", action="store_true", help="Also process recv_out (left channel) files from preFapMix with WhisperBite")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[mhrpTools] Processing {args.input} with mode '{args.mode}'...")
    mhrp_pipeline(args.input, args.output_dir, mode=args.mode, process_left_channel=args.process_left_channel)
    print(f"[mhrpTools] Done. Results in {args.output_dir}")

if __name__ == "__main__":
    main() 