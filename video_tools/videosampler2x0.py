import os
import random
import argparse
from moviepy.editor import VideoFileClip, concatenate_videoclips
from tqdm import tqdm
from datetime import datetime
import numpy as np

def read_log(log_file):
    used_segments = {}
    if log_file and os.path.exists(log_file):
        with open(log_file, 'r') as log:
            lines = log.readlines()
            current_file = None
            for line in lines:
                if line.startswith("Output file:"):
                    continue
                if line.startswith("Segments:"):
                    continue
                if line.startswith("File:"):
                    parts = line.split(',')
                    current_file = parts[0].split(': ')[1].strip()
                    start = float(parts[1].split(': ')[1].strip())
                    end = float(parts[2].split(': ')[1].strip())
                    if current_file not in used_segments:
                        used_segments[current_file] = []
                    used_segments[current_file].append((start, end))
    return used_segments

def is_segment_used(used_segments, filename, start, end):
    if filename in used_segments:
        for (used_start, used_end) in used_segments[filename]:
            if start < used_end and end > used_start:
                return True
    return False

def score_frame(frame):
    # Simple scoring function based on average brightness
    return np.mean(frame)

def create_sampler(input_dir, output_dir, duration, max_videos, segment_duration, filter_string, log_file, score_threshold):
    used_segments = read_log(log_file)

    video_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if filter_string:
        video_files = [f for f in video_files if filter_string in os.path.basename(f)]
    
    if not video_files:
        print("No video files found in the input directory.")
        return
    
    random.shuffle(video_files)  # Shuffle the list of video files
    
    if max_videos and max_videos < len(video_files):
        video_files = video_files[:max_videos]
    
    clips = []
    metadata = []
    total_clips_duration = 0
    target_duration = duration
    segment_duration = segment_duration or get_default_segment_duration(target_duration)

    for video_file in tqdm(video_files, desc="Processing videos"):
        clip = VideoFileClip(video_file)
        clip_duration = clip.duration
        filename = os.path.basename(video_file)

        if clip_duration > segment_duration:
            num_segments = int(clip_duration // segment_duration)
            attempts = 0

            while total_clips_duration < target_duration and attempts < num_segments:
                start = random.uniform(0, clip_duration - segment_duration)
                end = start + segment_duration

                if not is_segment_used(used_segments, filename, start, end):
                    subclip = clip.subclip(start, end)
                    frame = subclip.get_frame(0.5 * segment_duration)  # Take a frame from the middle of the segment
                    score = score_frame(frame)

                    if score >= score_threshold:
                        clips.append(subclip)
                        metadata.append((filename, start, end, score))
                        used_segments.setdefault(filename, []).append((start, end))
                        total_clips_duration += segment_duration
                attempts += 1
        else:
            start, end = 0, clip_duration
            if not is_segment_used(used_segments, filename, start, end):
                subclip = clip.subclip(start, end)
                frame = subclip.get_frame(0.5 * clip_duration)  # Take a frame from the middle of the segment
                score = score_frame(frame)

                if score >= score_threshold:
                    clips.append(subclip)
                    metadata.append((filename, start, end, score))
                    used_segments.setdefault(filename, []).append((start, end))
                    total_clips_duration += clip_duration

        if total_clips_duration >= target_duration:
            break

    random.shuffle(clips)  # Shuffle the collected clips to ensure randomness
    final_clip = concatenate_videoclips(clips[:int(target_duration // segment_duration)])  # Concatenate only the necessary number of clips
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = os.path.join(output_dir, f'sampler_{duration}_{timestamp}.mp4')
    final_clip.write_videofile(output_file, codec='libx264')
    
    if log_file:
        with open(log_file, 'a') as log:
            log.write(f"Output file: {output_file}\n")
            log.write("Segments:\n")
            for entry in metadata:
                log.write(f"File: {entry[0]}, Start: {entry[1]:.2f}, End: {entry[2]:.2f}, Score: {entry[3]:.2f}\n")
            log.write("\n")
    
    print(f'Sampler video saved as {output_file}')

def get_default_segment_duration(target_duration):
    return max(1, target_duration // 30)  # Default: 1 second for 30s output, 2s for 60s output, etc.

def main():
    parser = argparse.ArgumentParser(description='Create a sampler video from a folder of videos.')
    parser.add_argument('--input_dir', required=True, help='Path to the input directory containing video files.')
    parser.add_argument('--output_dir', required=True, help='Path to the output directory where sampler videos will be saved.')
    parser.add_argument('--max_videos', type=int, help='Maximum number of input videos to use for creating the sampler video.')
    parser.add_argument('--segment_duration', type=int, help='Duration of each segment in seconds.')
    parser.add_argument('--filter_string', type=str, help='Filter input files by a substring.')
    parser.add_argument('--log_file', type=str, help='Path to the log file for recording metadata.')
    parser.add_argument('--score_threshold', type=float, help='Minimum score for a segment to be included in the output.')

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    max_videos = args.max_videos
    segment_duration = args.segment_duration
    filter_string = args.filter_string
    log_file = args.log_file
    score_threshold = args.score_threshold or 0  # Default to 0 if no threshold is provided

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Creating 30-second sampler video...")
    create_sampler(input_dir, output_dir, 30, max_videos, segment_duration, filter_string, log_file, score_threshold)
    print("Creating 1-minute sampler video...")
    create_sampler(input_dir, output_dir, 60, max_videos, segment_duration, filter_string, log_file, score_threshold)

if __name__ == "__main__":
    main()
