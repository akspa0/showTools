import os
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Tuple
from tqdm import tqdm
from skimage.filters import sobel
from skimage.exposure import is_low_contrast
import multiprocessing

@dataclass
class FrameScore:
    frame_path: str
    timestamp: float
    overall_score: float
    brightness_score: float
    color_score: float
    edge_score: float
    blur_score: float

class FrameAnalyzer:
    @staticmethod
    def analyze_brightness(frame: np.ndarray) -> float:
        # Convert to grayscale if not already
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        mean_brightness = np.mean(gray)
        # Penalize too dark or too bright images
        optimal_brightness = 127
        brightness_score = 1 - (abs(mean_brightness - optimal_brightness) / 255)
        return max(0, brightness_score)

    @staticmethod
    def analyze_color_diversity(frame: np.ndarray) -> float:
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Calculate color histogram
        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        # Normalize histogram
        hist = cv2.normalize(hist, hist).flatten()
        # Calculate entropy as a measure of color diversity
        non_zero = hist[hist > 0]
        color_entropy = -np.sum(non_zero * np.log2(non_zero))
        # Normalize to 0-1 range (assuming max entropy for 180 bins is log2(180))
        return min(1.0, color_entropy / np.log2(180))

    @staticmethod
    def detect_edges(frame: np.ndarray) -> float:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculate edge magnitude using Sobel filter
        edge_magnitude = sobel(gray)
        # Normalize edge score
        edge_score = np.mean(edge_magnitude)
        return min(1.0, edge_score * 5)  # Adjust multiplier based on typical edge values

    @staticmethod
    def detect_blur(frame: np.ndarray) -> float:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculate variance of Laplacian
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize to 0-1 range (higher value means less blur)
        return min(1.0, laplacian_var / 500)  # Adjust denominator based on typical values

    @staticmethod
    def compute_overall_score(frame: np.ndarray) -> Tuple[float, float, float, float, float]:
        brightness_score = FrameAnalyzer.analyze_brightness(frame)
        color_score = FrameAnalyzer.analyze_color_diversity(frame)
        edge_score = FrameAnalyzer.detect_edges(frame)
        blur_score = FrameAnalyzer.detect_blur(frame)
        
        # Weight the components (adjust weights based on importance)
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
    def __init__(self, input_path: str, output_dir: str, frames_per_video: int = 10):
        self.input_path = input_path
        self.output_dir = output_dir
        self.frames_per_video = frames_per_video
        self.frame_scores: List[FrameScore] = []
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def process_video(self, video_path: str) -> List[FrameScore]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame interval to sample evenly throughout the video
        interval = max(1, total_frames // (self.frames_per_video * 2))
        frame_scores = []
        
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
                
                # Save frame if it meets minimum quality threshold
                if overall_score > 0.5:  # Adjust threshold as needed
                    frame_filename = f"{os.path.basename(video_path)}_{frame_count:08d}.jpg"
                    frame_path = os.path.join(self.output_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    
                    frame_scores.append(FrameScore(
                        frame_path=frame_path,
                        timestamp=timestamp,
                        overall_score=overall_score,
                        brightness_score=brightness_score,
                        color_score=color_score,
                        edge_score=edge_score,
                        blur_score=blur_score
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
        
        # Use ProcessPoolExecutor for parallel processing
        num_processes = max(1, multiprocessing.cpu_count() - 1)
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(self.process_video, video_files))
        
        # Flatten results and sort by score
        all_scores = [score for video_scores in results for score in video_scores]
        all_scores.sort(key=lambda x: x.overall_score, reverse=True)
        
        return all_scores

    def save_best_frames(self, scores: List[FrameScore], max_frames: int = 100):
        best_frames_dir = os.path.join(self.output_dir, 'best_frames')
        if not os.path.exists(best_frames_dir):
            os.makedirs(best_frames_dir)
        
        # Take top N frames
        best_scores = scores[:max_frames]
        
        print(f"\nSaving top {len(best_scores)} frames...")
        for i, score in enumerate(best_scores):
            # Copy frame to best_frames directory with score in filename
            new_filename = f"frame_{i:04d}_score_{score.overall_score:.3f}.jpg"
            new_path = os.path.join(best_frames_dir, new_filename)
            
            try:
                img = Image.open(score.frame_path)
                img.save(new_path, quality=95)
                print(f"Saved {new_filename} (Score: {score.overall_score:.3f})")
            except Exception as e:
                print(f"Error saving {new_filename}: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Advanced Video Frame Sampler with Aesthetic Scoring')
    parser.add_argument('--input_dir', required=True, help='Input directory containing video files')
    parser.add_argument('--output_dir', required=True, help='Output directory for sampled frames')
    parser.add_argument('--frames_per_video', type=int, default=10, help='Number of frames to sample per video')
    parser.add_argument('--max_output_frames', type=int, default=100, help='Maximum number of best frames to save')
    
    args = parser.parse_args()
    
    processor = VideoProcessor(
        input_path=args.input_dir,
        output_dir=args.output_dir,
        frames_per_video=args.frames_per_video
    )
    
    print(f"Processing videos in {args.input_dir}...")
    frame_scores = processor.process_videos_parallel()
    
    if frame_scores:
        processor.save_best_frames(frame_scores, args.max_output_frames)
        print("\nProcessing complete!")
    else:
        print("No frames were processed.")

if __name__ == "__main__":
    main()