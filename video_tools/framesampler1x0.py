import os
import random
import argparse
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from datetime import datetime

def sample_frames(input_dir, output_dir, num_frames):
    video_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print("No video files found in the input directory.")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        clip = VideoFileClip(video_file)
        clip_duration = clip.duration
        filename = os.path.basename(video_file)
        
        for i in range(num_frames):
            frame_time = random.uniform(0, clip_duration)
            frame = clip.get_frame(frame_time)
            frame_filename = os.path.join(output_dir, f"{i:04d}.png")
            frame_image = Image.fromarray(frame)
            frame_image.save(frame_filename)
    
    print(f"Sampled frames saved in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Sample random frames from video files and save as PNG images.')
    parser.add_argument('--input_dir', required=True, help='Path to the input directory containing video files.')
    parser.add_argument('--output_dir', required=True, help='Path to the output directory where sampled frames will be saved.')
    parser.add_argument('--num_frames', type=int, default=1000, help='Number of random frames to sample from each video file.')

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    num_frames = args.num_frames

    sample_frames(input_dir, output_dir, num_frames)

if __name__ == "__main__":
    main()
