import os
import argparse
from pydub import AudioSegment
from datetime import datetime

def log_mismatch(log_file, trans_file, recv_file, trans_length, recv_length):
    with open(log_file, 'a') as log:
        log.write(f"{datetime.now()}: Length mismatch - {trans_file} ({trans_length} ms) and {recv_file} ({recv_length} ms)\n")

def process_audio_files(input_folder, output_folder, log_file):
    for root, _, files in os.walk(input_folder):
        trans_files = [f for f in files if f.startswith("trans_out-") and f.endswith(".wav")]
        recv_files = [f for f in files if f.startswith("recv_out-") and f.endswith(".wav")]
        
        for trans_file in trans_files:
            suffix = trans_file[len("trans_out-"):]
            trans_file_path = os.path.join(root, trans_file)
            recv_file_path = os.path.join(root, f"recv_out-{suffix}")
            
            if recv_file_path in [os.path.join(root, f) for f in recv_files]:
                trans_audio = AudioSegment.from_wav(trans_file_path)
                recv_audio = AudioSegment.from_wav(recv_file_path)
                
                # Check for length mismatch and log it
                if len(trans_audio) != len(recv_audio):
                    log_mismatch(log_file, trans_file_path, recv_file_path, len(trans_audio), len(recv_audio))
                    continue  # Skip this pair if lengths don't match
                
                # Swap channels: trans_audio to left, recv_audio to right
                stereo_audio = AudioSegment.from_mono_audiosegments(recv_audio, trans_audio)
                
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                
                stereo_file_path = os.path.join(output_dir, f"stereo_out-{suffix}")
                stereo_audio.export(stereo_file_path, format="wav")
                print(f"Processed {trans_file} and {recv_file_path} into {stereo_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Process audio files into stereo output.")
    parser.add_argument("input_folder", help="Path to the input folder containing audio files.")
    parser.add_argument("output_folder", help="Path to the output folder where processed files will be saved.")
    parser.add_argument("log_file", help="Path to the log file for recording mismatches.")
    
    args = parser.parse_args()
    
    process_audio_files(args.input_folder, args.output_folder, args.log_file)

if __name__ == "__main__":
    main()
