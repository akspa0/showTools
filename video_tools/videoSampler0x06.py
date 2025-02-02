import os
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Tuple, Optional
from tqdm import tqdm
from skimage.filters import sobel
from skimage.exposure import is_low_contrast
import multiprocessing
import re
import shutil

@dataclass
class FrameScore:
    frame_path: str
    timestamp: float
    overall_score: float
    brightness_score: float
    color_score: float
    edge_score: float
    blur_score: float
    year: Optional[int] = None

class YearExtractor:
    @staticmethod
    def extract_year(filename: str) -> Optional[int]:
        # Pattern to match years between 1900 and 2099
        year_pattern = r'(?:19|20)\d{2}'
        matches = re.findall(year_pattern, filename)
        
        if matches:
            # Return the first valid year found
            year = int(matches[0])
            if 1900 <= year <= 2099:  # Validate year range
                return year
        return None

class FrameAnalyzer:
    @staticmethod
    def analyze_brightness(frame: np.ndarray) -> float:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        mean_brightness = np.mean(gray)
        optimal_brightness = 127
        brightness_score = 1 - (abs(mean_brightness - optimal_brightness) / 255)
        return max(0, brightness_score)

    @staticmethod
    def analyze_color_diversity(frame: np.ndarray) -> float:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist = cv2.normalize(hist, hist).flatten()
        non_zero = hist[hist > 0]
        color_entropy = -np.sum(non_zero * np.log2(non_zero))
        return min(1.0, color_entropy / np.log2(180))

    @staticmethod
    def detect_edges(frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edge_magnitude = sobel(gray)
        edge_score = np.mean(edge_magnitude)
        return min(1.0, edge_score * 5)

    @staticmethod
    def detect_blur(frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return min(1.0, laplacian_var / 500)

    @staticmethod
    def compute_overall_score(frame: np.ndarray) -> Tuple[float, float, float, float, float]:
        brightness_score = FrameAnalyzer.analyze_brightness(frame)
        color_score = FrameAnalyzer.analyze_color_diversity(frame)
        edge_score = FrameAnalyzer.detect_edges(frame)
        blur_score = FrameAnalyzer.detect_blur(frame)
        
        weights = {
            'brightness': 0.25,
            'color': 0.25,
            'edge': 0.25,
            'blur': 0.25
        }
        
        overall_score = (
            brightness_score * weights['brightness'] +
            color_score * weights['color'] +
            edge_score * weights['edge'] +
            blur_score * weights['blur']
        )
        
        return overall_score, brightness_score, color_score, edge_score, blur_score

class VideoProcessor:
    def __init__(self, input_path: str, output_dir: str, frames_per_video: int = 10, aesthetic_threshold: float = 0.75):
        self.input_path = input_path
        self.output_dir = output_dir
        self.frames_per_video = frames_per_video
        self.aesthetic_threshold = aesthetic_threshold
        self.frame_scores: List[FrameScore] = []
        
        # Create output directory structure
        self.base_output_dir = os.path.join(output_dir, 'all_frames')
        self.aesthetic_dir = os.path.join(output_dir, f'{aesthetic_threshold:.2f}_aesthetic')
        
        for directory in [self.base_output_dir, self.aesthetic_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def process_video(self, video_path: str) -> List[FrameScore]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = max(1, total_frames // (self.frames_per_video * 2))
        frame_scores = []
        
        # Extract year from video filename
        video_year = YearExtractor.extract_year(os.path.basename(video_path))
        
        with tqdm(total=self.frames_per_video, desc=f"Processing {os.path.basename(video_path)}") as pbar:
            frame_count = 0
            while len(frame_scores) < self.frames_per_video and frame_count < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                timestamp = frame_count / fps
                overall_score, brightness_score, color_score, edge_score, blur_score = (
                    FrameAnalyzer.compute_overall_score(frame)
                )
                
                frame_filename = f"{os.path.basename(video_path)}_{frame_count:08d}.jpg"
                frame_path = os.path.join(self.base_output_dir, frame_filename)
                
                # Save frame and create FrameScore object
                cv2.imwrite(frame_path, frame)
                frame_scores.append(FrameScore(
                    frame_path=frame_path,
                    timestamp=timestamp,
                    overall_score=overall_score,
                    brightness_score=brightness_score,
                    color_score=color_score,
                    edge_score=edge_score,
                    blur_score=blur_score,
                    year=video_year
                ))
                pbar.update(1)
                
                frame_count += interval
        
        cap.release()
        return frame_scores

    def process_videos_parallel(self) -> List[FrameScore]:
        video_files = [
            os.path.join(self.input_path, f) for f in os.listdir(self.input_path)
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ]
        
        if not video_files:
            print("No video files found in the input directory.")
            return []
        
        num_processes = max(1, multiprocessing.cpu_count() - 1)
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(self.process_video, video_files))
        
        all_scores = [score for video_scores in results for score in video_scores]
        all_scores.sort(key=lambda x: x.overall_score, reverse=True)
        
        return all_scores

    def organize_frames_by_year(self, scores: List[FrameScore]):
        # Create year directories in both base and aesthetic directories
        years_dict = {}
        for score in scores:
            if score.year:
                if score.year not in years_dict:
                    years_dict[score.year] = []
                years_dict[score.year].append(score)
        
        # Create year directories and organize frames
        for year, year_scores in years_dict.items():
            # Create year directories
            base_year_dir = os.path.join(self.base_output_dir, str(year))
            aesthetic_year_dir = os.path.join(self.aesthetic_dir, str(year))
            
            os.makedirs(base_year_dir, exist_ok=True)
            os.makedirs(aesthetic_year_dir, exist_ok=True)
            
            # Move files to appropriate directories
            for score in year_scores:
                # Move to year-specific directory in base_output_dir
                new_base_path = os.path.join(base_year_dir, os.path.basename(score.frame_path))
                shutil.move(score.frame_path, new_base_path)
                score.frame_path = new_base_path
                
                # If meets aesthetic threshold, copy to aesthetic directory
                if score.overall_score >= self.aesthetic_threshold:
                    aesthetic_filename = f"score_{score.overall_score:.3f}_{os.path.basename(score.frame_path)}"
                    aesthetic_path = os.path.join(aesthetic_year_dir, aesthetic_filename)
                    shutil.copy2(new_base_path, aesthetic_path)

    def process_and_organize(self):
        print(f"Processing videos in {self.input_path}...")
        frame_scores = self.process_videos_parallel()
        
        if frame_scores:
            print("\nOrganizing frames by year and aesthetic score...")
            self.organize_frames_by_year(frame_scores)
            
            # Print summary
            aesthetic_count = sum(1 for score in frame_scores if score.overall_score >= self.aesthetic_threshold)
            print(f"\nProcessing complete!")
            print(f"Total frames processed: {len(frame_scores)}")
            print(f"Frames meeting aesthetic threshold ({self.aesthetic_threshold:.2f}): {aesthetic_count}")
            
            # Print year-wise statistics
            years = set(score.year for score in frame_scores if score.year is not None)
            for year in sorted(years):
                year_frames = [s for s in frame_scores if s.year == year]
                year_aesthetic = sum(1 for s in year_frames if s.overall_score >= self.aesthetic_threshold)
                print(f"\nYear {year}:")
                print(f"  Total frames: {len(year_frames)}")
                print(f"  Aesthetic frames: {year_aesthetic}")
        else:
            print("No frames were processed.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Advanced Video Frame Sampler with Aesthetic Scoring and Year Organization')
    parser.add_argument('--input_dir', required=True, help='Input directory containing video files')
    parser.add_argument('--output_dir', required=True, help='Output directory for sampled frames')
    parser.add_argument('--frames_per_video', type=int, default=10, help='Number of frames to sample per video')
    parser.add_argument('--aesthetic_threshold', type=float, default=0.75, help='Threshold for aesthetic frame selection')
    
    args = parser.parse_args()
    
    processor = VideoProcessor(
        input_path=args.input_dir,
        output_dir=args.output_dir,
        frames_per_video=args.frames_per_video,
        aesthetic_threshold=args.aesthetic_threshold
    )
    
    processor.process_and_organize()

if __name__ == "__main__":
    main()