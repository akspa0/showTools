#!/usr/bin/env python3

import os
import argparse
import shutil
import subprocess

def convert_audio_to_flac(input_file, output_file):
    """
    Convert .wav or .au to .flac using ffmpeg with error handling.
    """
    cmd = [
        "ffmpeg",
        "-err_detect", "ignore_err",
        "-i", input_file,
        "-y",
        "-loglevel", "warning",
        output_file
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        # Remove partial output if ffmpeg wrote anything
        if os.path.exists(output_file):
            os.remove(output_file)
        raise e

def convert_flac_to_format(input_file, output_file):
    """
    Convert .flac back to either .wav or .au (determined by output_file extension).
    """
    cmd = [
        "ffmpeg",
        "-err_detect", "ignore_err",
        "-i", input_file,
        "-y",
        "-loglevel", "warning",
        output_file
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        if os.path.exists(output_file):
            os.remove(output_file)
        raise e

def main():
    parser = argparse.ArgumentParser(
        description="Convert .wav/.au -> .flac, or restore .flac -> .wav/.au, preserving folder structure."
    )
    parser.add_argument("source", help="Path to the source folder")
    parser.add_argument("destination", help="Path to the destination folder")
    parser.add_argument(
        "--mode",
        choices=["convert", "restore"],
        default="convert",
        help="Mode: 'convert' (wav/au->flac) or 'restore' (flac->wav/au). Default='convert'."
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.5,
        help="Estimated compression ratio for (wav/au)->flac (0.0 to 1.0). Default=0.5"
    )
    parser.add_argument(
        "--restore-format",
        choices=["wav", "au"],
        default="wav",
        help="When restoring flac->?, output file format extension. Default='wav'."
    )
    args = parser.parse_args()

    source_folder = os.path.abspath(args.source)
    destination_folder = os.path.abspath(args.destination)

    if not os.path.exists(source_folder):
        print(f"[ERROR] Source folder does not exist: {source_folder}")
        return

    os.makedirs(destination_folder, exist_ok=True)

    if args.mode == "convert":
        # 1) Calculate total size of .wav + .au files & estimate flac size
        total_input_size = 0
        for root, _, files in os.walk(source_folder):
            for f in files:
                ext = os.path.splitext(f.lower())[1]
                if ext in [".wav", ".au"]:
                    filepath = os.path.join(root, f)
                    try:
                        total_input_size += os.path.getsize(filepath)
                    except OSError:
                        pass  # skip unreadable or locked files

        if total_input_size == 0:
            print("[INFO] No .wav or .au files found in source folder. Nothing to convert.")
            return

        estimated_flac_size = int(total_input_size * args.ratio)
        savings = total_input_size - estimated_flac_size

        print("==== Estimated Space Usage ====")
        print(f"Total .wav/.au size (on disk): {total_input_size/1_000_000:.2f} MB")
        print(f"Estimated total .flac size:    {estimated_flac_size/1_000_000:.2f} MB")
        print(f"Approx. space savings:        {savings/1_000_000:.2f} MB")
        print("================================\n")

        confirm = input("Proceed with conversion? (y/n): ").strip().lower()
        if confirm not in ["y", "yes"]:
            print("[ABORT] User chose not to proceed.")
            return

        # 2) Perform actual conversion / copying
        for root, _, files in os.walk(source_folder):
            rel_path = os.path.relpath(root, source_folder)
            dest_path = os.path.join(destination_folder, rel_path)
            os.makedirs(dest_path, exist_ok=True)

            for f in files:
                source_file = os.path.join(root, f)
                ext = os.path.splitext(f.lower())[1]

                if ext in [".wav", ".au"]:
                    # Convert these to .flac
                    out_filename = os.path.splitext(f)[0] + ".flac"
                    out_file = os.path.join(dest_path, out_filename)
                    print(f"[CONVERT] {ext} -> FLAC:\n  {source_file}\n  -> {out_file}")
                    try:
                        convert_audio_to_flac(source_file, out_file)
                    except Exception as e:
                        print(f"[ERROR] Could not convert {source_file}. Skipping. Reason: {e}")
                        continue
                else:
                    # Copy everything else as-is (.mp3, .txt, .jpg, etc.)
                    out_file = os.path.join(dest_path, f)
                    print(f"[COPY] {source_file} -> {out_file}")
                    try:
                        shutil.copy2(source_file, out_file)
                    except Exception as e:
                        print(f"[ERROR] Could not copy {source_file}. Skipping. Reason: {e}")
                        continue

        print("[DONE] Conversion complete.")

    else:  # args.mode == "restore"
        # If .flac files are found, convert them back to either .wav or .au
        # Copy all other files as-is.
        restore_ext = "." + args.restore_format  # e.g., ".wav" or ".au"

        for root, _, files in os.walk(source_folder):
            rel_path = os.path.relpath(root, source_folder)
            dest_path = os.path.join(destination_folder, rel_path)
            os.makedirs(dest_path, exist_ok=True)

            for f in files:
                source_file = os.path.join(root, f)
                ext = os.path.splitext(f.lower())[1]

                if ext == ".flac":
                    out_filename = os.path.splitext(f)[0] + restore_ext
                    out_file = os.path.join(dest_path, out_filename)
                    print(f"[RESTORE] FLAC -> {restore_ext}:\n  {source_file}\n  -> {out_file}")
                    try:
                        convert_flac_to_format(source_file, out_file)
                    except Exception as e:
                        print(f"[ERROR] Could not restore {source_file}. Skipping. Reason: {e}")
                        continue
                else:
                    # Copy everything else as-is
                    out_file = os.path.join(dest_path, f)
                    print(f"[COPY] {source_file} -> {out_file}")
                    try:
                        shutil.copy2(source_file, out_file)
                    except Exception as e:
                        print(f"[ERROR] Could not copy {source_file}. Skipping. Reason: {e}")
                        continue

        print("[DONE] Restore complete.")

if __name__ == "__main__":
    main()
