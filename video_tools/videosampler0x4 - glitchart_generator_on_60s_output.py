import os
import random
import argparse
from moviepy.editor import VideoFileClip, concatenate_videoclips
from tqdm import tqdm

def create_sampler(input_dir, output_dir, duration, max_videos, segment_duration):
    video_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print("No video files found in the input directory.")
        return
    
    random.shuffle(video_files)  # Shuffle the list of video files
    
    if max_videos and max_videos < len(video_files):
        video_files = video_files[:max_videos]
    
    clips = []
    total_clips_duration = 0
    target_duration = duration

    for video_file in tqdm(video_files, desc="Processing videos"):
        clip = VideoFileClip(video_file)
        clip_duration = clip.duration

        if clip_duration <= segment_duration:
            subclip = clip.subclip(0, clip_duration)
            clips.append(subclip)
            total_clips_duration += clip_duration
        else:
            num_segments = int(clip_duration // segment_duration)
            segment_starts = random.sample(range(0, int(clip_duration - segment_duration)), min(num_segments, int((target_duration - total_clips_duration) // segment_duration)))

            for start in segment_starts:
                subclip = clip.subclip(start, start + segment_duration)
                clips.append(subclip)
                total_clips_duration += segment_duration

                if total_clips_duration >= target_duration:
                    break
        if total_clips_duration >= target_duration:
            break

    final_clip = concatenate_videoclips(clips)
    output_file = os.path.join(output_dir, f'sampler_{duration}.mp4')
    final_clip.write_videofile(output_file, codec='libx264')
    print(f'Sampler video saved as {output_file}')

def main():
    parser = argparse.ArgumentParser(description='Create a sampler video from a folder of videos.')
    parser.add_argument('--input_dir', required=True, help='Path to the input directory containing video files.')
    parser.add_argument('--output_dir', required=True, help='Path to the output directory where sampler videos will be saved.')
    parser.add_argument('--max_videos', type=int, help='Maximum number of input videos to use for creating the sampler video.')
    parser.add_argument('--segment_duration', type=int, help='Duration of each segment in seconds.')

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    max_videos = args.max_videos
    segment_duration = args.segment_duration

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def get_default_segment_duration(target_duration):
        return max(1, target_duration // 30)  # Default: 1 second for 30s output, 2s for 60s output, etc.

    segment_duration_30s = segment_duration if segment_duration else get_default_segment_duration(30)
    segment_duration_60s = segment_duration if segment_duration else get_default_segment_duration(60)

    print("Creating 30-second sampler video...")
    create_sampler(input_dir, output_dir, 30, max_videos, segment_duration_30s)
    print("Creating 1-minute sampler video...")
    create_sampler(input_dir, output_dir, 60, max_videos, segment_duration_60s)

if __name__ == "__main__":
    main()
