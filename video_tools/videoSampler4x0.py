import os
import random
import argparse
from itertools import combinations
from moviepy.editor import VideoFileClip, concatenate_videoclips
from tqdm import tqdm
from PIL import Image
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

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

def filter_videos(video_files, keywords_list):
    for r in range(len(keywords_list), 0, -1):
        for combo in combinations(keywords_list, r):
            filtered_files = [f for f in video_files if all(keyword in os.path.basename(f) for keyword in combo)]
            if len(filtered_files) >= 2:
                return filtered_files
    return []

def search_frames(input_dir, output_dir, max_videos, segment_duration, keywords, log_file):
    used_segments = read_log(log_file)

    video_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if keywords:
        keywords_list = [keyword.strip() for keyword in keywords.split(',')]
        video_files = filter_videos(video_files, keywords_list)
    
    if not video_files:
        print("No video files found in the input directory.")
        return
    
    random.shuffle(video_files)  # Shuffle the list of video files
    
    if max_videos and max_videos < len(video_files):
        video_files = video_files[:max_videos]
    
    metadata = []
    segment_duration = segment_duration or get_default_segment_duration(60)

    for video_file in tqdm(video_files, desc="Processing videos"):
        clip = VideoFileClip(video_file)
        clip_duration = clip.duration
        filename = os.path.basename(video_file)

        if clip_duration > segment_duration:
            num_segments = int(clip_duration // segment_duration)
            attempts = 0

            while attempts < num_segments:
                start = random.uniform(0, clip_duration - segment_duration)
                end = start + segment_duration

                if not is_segment_used(used_segments, filename, start, end):
                    metadata.append((filename, start, end))
                    used_segments.setdefault(filename, []).append((start, end))
                attempts += 1
        else:
            start, end = 0, clip_duration
            if not is_segment_used(used_segments, filename, start, end):
                metadata.append((filename, start, end))
                used_segments.setdefault(filename, []).append((start, end))

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    metadata_file = os.path.join(output_dir, f'metadata_{timestamp}.txt')
    with open(metadata_file, 'w') as f:
        for entry in metadata:
            f.write(f"File: {entry[0]}, Start: {entry[1]:.2f}, End: {entry[2]:.2f}\n")
    
    print(f'Metadata saved as {metadata_file}')
    return metadata_file

def create_sampler(input_dir, output_dir, duration, segment_duration, metadata_file, glitch=False):
    segment_duration = segment_duration or get_default_segment_duration(duration)
    
    with open(metadata_file, 'r') as f:
        metadata = [line.strip().split(', ') for line in f.readlines()]

    clips = []
    total_clips_duration = 0
    target_duration = duration

    for entry in tqdm(metadata, desc="Processing metadata"):
        try:
            filename = entry[0].split(': ')[1]
            start = float(entry[1].split(': ')[1])
            end = float(entry[2].split(': ')[1])
            video_file = os.path.join(input_dir, filename)

            clip = VideoFileClip(video_file)
            subclip = clip.subclip(start, end)
            clips.append(subclip)
            total_clips_duration += (end - start)

            if total_clips_duration >= target_duration:
                break
        except (IndexError, ValueError) as e:
            print(f"Error parsing metadata entry {entry}: {e}")
        except Exception as e:
            print(f"Error processing {video_file} from {start} to {end}: {e}")

    if glitch:
        random.shuffle(clips)  # Shuffle the collected clips to ensure randomness

    try:
        final_clip = concatenate_videoclips(clips[:int(target_duration // segment_duration)])  # Concatenate only the necessary number of clips
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_file = os.path.join(output_dir, f'sampler_{duration}_{timestamp}.mp4')
        final_clip.write_videofile(output_file, codec='libx264')
        print(f'Sampler video saved as {output_file}')
        return output_file
    except Exception as e:
        print(f"Error creating sampler video: {e}")
        return None

def sample_frames_from_video(video_file, output_dir, num_frames):
    clip = VideoFileClip(video_file)
    clip_duration = clip.duration
    fps = clip.fps
    num_frames_in_clip = int(clip_duration * fps)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    selected_frames = set()
    while len(selected_frames) < num_frames:
        frame_index = random.randint(0, num_frames_in_clip - 1)
        if frame_index not in selected_frames:
            selected_frames.add(frame_index)

    def save_frame(frame_index):
        try:
            frame_time = frame_index / fps
            frame = clip.get_frame(frame_time)
            frame_filename = os.path.join(output_dir, f"{frame_index:08d}.png")  # Ensure unique filenames
            frame_image = Image.fromarray(frame)
            frame_image.save(frame_filename)
            print(f"Saved frame {frame_index} from {video_file} at time {frame_time:.2f}s")
        except Exception as e:
            print(f"Error saving frame {frame_index} from {video_file} at time {frame_time:.2f}s: {e}")

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(save_frame, selected_frames), total=len(selected_frames), desc="Saving frames"))

    print(f"Sampled frames saved in {output_dir}")

def get_default_segment_duration(target_duration):
    return max(1, target_duration // 30)  # Default: 1 second for 30s output, 2s for 60s output, etc.

def main():
    parser = argparse.ArgumentParser(description='Search for frames based on keywords, save metadata, create a sampler video, and sample random frames from the resulting output.')
    parser.add_argument('--input_dir', required=True, help='Path to the input directory containing video files.')
    parser.add_argument('--output_dir', required=True, help='Path to the output directory where metadata, sampler videos, and sampled frames will be saved.')
    parser.add_argument('--max_videos', type=int, help='Maximum number of input videos to process.')
    parser.add_argument('--segment_duration', type=int, help='Duration of each segment in seconds.')
    parser.add_argument('--keywords', type=str, help='Comma-separated list of keywords to filter input files by their filenames.')
    parser.add_argument('--log_file', type=str, help='Path to the log file for recording metadata.')
    parser.add_argument('--duration', type=int, required=True, help='Duration of the sampler video in seconds.')
    parser.add_argument('--num_frames', type=int, default=1000, help='Number of random frames to sample from the video files.')
    parser.add_argument('--glitch', action='store_true', help='Enable glitch mode for artistic video generation.')

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    max_videos = args.max_videos
    segment_duration = args.segment_duration
    keywords = args.keywords
    log_file = args.log_file
    duration = args.duration
    num_frames = args.num_frames
    glitch = args.glitch

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metadata_file = search_frames(input_dir, output_dir, max_videos, segment_duration, keywords, log_file)
    if metadata_file:
        sampler_video = create_sampler(input_dir, output_dir, duration, segment_duration, metadata_file, glitch)
        if sampler_video:
            sample_frames_from_video(sampler_video, output_dir, num_frames)

if __name__ == "__main__":
    main()
