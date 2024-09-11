import os
import random
import argparse
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def sample_frames(input_dir, output_dir, num_frames, keywords, max_files, max_duration):
    video_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    # Filter video files based on keywords
    if keywords:
        keywords_list = [keyword.strip() for keyword in keywords.split(',')]
        video_files = [f for f in video_files if all(keyword in os.path.basename(f) for keyword in keywords_list)]
    
    # Limit the number of input files
    if max_files:
        video_files = video_files[:max_files]
    
    if not video_files:
        print("No video files found in the input directory matching the keywords.")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    total_frames = 0
    frame_times = []
    video_clips = []

    # Calculate total number of frames and collect frame times
    for video_file in video_files:
        clip = VideoFileClip(video_file)
        clip_duration = clip.duration
        if clip_duration > max_duration:
            print(f"Skipping {video_file} (duration: {clip_duration:.2f}s) - exceeds max duration of {max_duration:.2f}s")
            continue
        fps = clip.fps
        num_frames_in_clip = int(clip_duration * fps)
        total_frames += num_frames_in_clip
        video_clips.append((clip, num_frames_in_clip))
        print(f"Processing {video_file}: {num_frames_in_clip} frames")

    if num_frames > total_frames:
        num_frames = total_frames

    selected_frames = set()
    while len(selected_frames) < num_frames:
        frame_index = random.randint(0, total_frames - 1)
        if frame_index not in selected_frames:
            selected_frames.add(frame_index)

    def save_frame(frame_index):
        cumulative_frames = 0
        for clip, num_frames_in_clip in video_clips:
            if cumulative_frames + num_frames_in_clip > frame_index:
                frame_time = (frame_index - cumulative_frames) / clip.fps
                frame = clip.get_frame(frame_time)
                frame_filename = os.path.join(output_dir, f"{frame_index:08d}.png")  # Ensure unique filenames
                frame_image = Image.fromarray(frame)
                frame_image.save(frame_filename)
                print(f"Saved frame {frame_index} from {clip.filename} at time {frame_time:.2f}s")
                break
            cumulative_frames += num_frames_in_clip

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(save_frame, selected_frames), total=len(selected_frames), desc="Saving frames"))

    print(f"Sampled frames saved in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Sample random frames from video files and save as PNG images.')
    parser.add_argument('--input_dir', required=True, help='Path to the input directory containing video files.')
    parser.add_argument('--output_dir', required=True, help='Path to the output directory where sampled frames will be saved.')
    parser.add_argument('--num_frames', type=int, default=1000, help='Number of random frames to sample from the video files.')
    parser.add_argument('--keywords', type=str, help='Comma-separated list of keywords to filter input files by their filenames.')
    parser.add_argument('--max_files', type=int, help='Maximum number of input files to process.')
    parser.add_argument('--max_duration', type=float, default=300, help='Maximum duration (in seconds) of input video files to process.')

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    num_frames = args.num_frames
    keywords = args.keywords
    max_files = args.max_files
    max_duration = args.max_duration

    sample_frames(input_dir, output_dir, num_frames, keywords, max_files, max_duration)

if __name__ == "__main__":
    main()
