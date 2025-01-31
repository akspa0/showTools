import os
import sys
import argparse
from file_handler import (
    search_files, find_matching_audio, get_relative_path,
    create_output_structure, copy_support_files
)
from cue_parser import parse_cue
from audio_splitter import split_audio
from error_handler import (
    handle_missing_file, handle_mismatched_files,
    handle_encoding_error, handle_processing_error
)

def process_directory(input_dir, output_base=None):
    """Process a single directory containing CUE and audio files."""
    # Search for all relevant files
    cue_files, audio_files, artwork_files, log_files = search_files(input_dir)

    if not cue_files:
        handle_missing_file("cue files")
        return False

    if not audio_files:
        handle_missing_file("audio files")
        return False

    try:
        # Determine output directory while maintaining structure
        if output_base:
            output_dir, artwork_dir = create_output_structure(input_dir, output_base)
        else:
            output_dir = os.path.join(input_dir, 'split')
            artwork_dir = os.path.join(output_dir, 'artwork')
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(artwork_dir, exist_ok=True)

        # Copy artwork and log files
        if output_base:
            copy_support_files(artwork_files + log_files, input_dir, output_base)

        # Process each CUE file
        for cue_file in cue_files:
            try:
                cue_data = parse_cue(cue_file)
                matching_audio_file = find_matching_audio(cue_file, audio_files)
                if not matching_audio_file:
                    handle_mismatched_files(cue_file, audio_files)
                    continue
                
                print(f"\nProcessing album: {cue_data[0]['album_title']}")
                print(f"Artist: {cue_data[0]['album_artist']}")
                split_audio(cue_data, matching_audio_file, output_dir)
            except ValueError as e:
                if "Could not decode" in str(e):
                    handle_encoding_error(cue_file, ['utf-8', 'cp1252', 'iso-8859-1', 'latin1'])
                else:
                    handle_processing_error(cue_file, e)
                continue
            except Exception as e:
                handle_processing_error(cue_file, e)
                continue

        return True
    except Exception as e:
        handle_processing_error(input_dir, e)
        return False

def process_bulk(input_base, output_base):
    """Process all subdirectories containing CUE and audio files."""
    success = True
    
    # First, create the base output directory
    os.makedirs(output_base, exist_ok=True)
    
    # Walk through all directories
    for root, _, files in os.walk(input_base):
        # Check if directory contains CUE files
        if any(f.lower().endswith('.cue') for f in files):
            print(f"\nProcessing directory: {root}")
            if not process_directory(root, output_base):
                success = False
                print(f"Failed to process directory: {root}")
            
    return success

def main():
    parser = argparse.ArgumentParser(description='Split audio files using CUE sheets.')
    parser.add_argument('input', help='Input directory containing CUE and audio files')
    parser.add_argument('-o', '--output', help='Output base directory for bulk processing')
    parser.add_argument('-b', '--bulk', action='store_true',
                       help='Process all subdirectories recursively')
    
    args = parser.parse_args()

    if args.bulk:
        if not args.output:
            print("Error: Output directory (-o) is required for bulk processing")
            sys.exit(1)
        success = process_bulk(args.input, args.output)
    else:
        success = process_directory(args.input, args.output)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()