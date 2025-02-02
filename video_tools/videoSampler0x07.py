import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Tuple, Optional
from tqdm import tqdm
import multiprocessing
import re
import shutil
from torchvision.models import resnet18
from torch.cuda.amp import autocast

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
        year_pattern = r'(?:19|20)\d{2}'
        matches = re.findall(year_pattern, filename)
        
        if matches:
            year = int(matches[0])
            if 1900 <= year <= 2099:
                return year
        return None

class FrameAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        # Load pre-trained ResNet for feature extraction
        self.feature_extractor = resnet18(pretrained=True).to(self.device)
        self.feature_extractor.eval()
        
        # Create edge detection kernels as PyTorch tensors
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(self.device)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).to(self.device)
        self.sobel_x = self.sobel_x.view(1, 1, 3, 3)
        self.sobel_y = self.sobel_y.view(1, 1, 3, 3)

    @torch.no_grad()
    def analyze_brightness(self, frame_tensor: torch.Tensor) -> float:
        # Convert to grayscale using GPU
        gray = 0.2989 * frame_tensor[0] + 0.5870 * frame_tensor[1] + 0.1140 * frame_tensor[2]
        mean_brightness = gray.mean().item()
        optimal_brightness = 0.5  # Normalized to [0, 1]
        brightness_score = 1 - abs(mean_brightness - optimal_brightness) * 2
        return max(0, brightness_score)

    @torch.no_grad()
    def analyze_color_diversity(self, frame_tensor: torch.Tensor) -> float:
        # Calculate color histogram on GPU
        hist = torch.histc(frame_tensor, bins=64, min=0, max=1)
        hist = hist / hist.sum()
        non_zero = hist[hist > 0]
        color_entropy = -(non_zero * torch.log2(non_zero)).sum().item()
        return min(1.0, color_entropy / 6.0)  # Normalize (log2(64) ≈ 6)

    @torch.no_grad()
    def detect_edges(self, frame_tensor: torch.Tensor) -> float:
        # Convert to grayscale
        gray = 0.2989 * frame_tensor[0] + 0.5870 * frame_tensor[1] + 0.1140 * frame_tensor[2]
        gray = gray.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        # Apply Sobel filters using convolution
        gx = nn.functional.conv2d(gray, self.sobel_x, padding=1)
        gy = nn.functional.conv2d(gray, self.sobel_y, padding=1)
        
        # Calculate gradient magnitude
        edge_magnitude = torch.sqrt(gx ** 2 + gy ** 2)
        edge_score = edge_magnitude.mean().item()
        return min(1.0, edge_score * 5)

    @torch.no_grad()
    def detect_blur(self, frame_tensor: torch.Tensor) -> float:
        # Convert to grayscale
        gray = 0.2989 * frame_tensor[0] + 0.5870 * frame_tensor[1] + 0.1140 * frame_tensor[2]
        gray = gray.unsqueeze(0).unsqueeze(0)
        
        # Laplacian kernel
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).to(self.device).view(1, 1, 3, 3)
        
        # Apply Laplacian filter
        laplacian = nn.functional.conv2d(gray, laplacian_kernel, padding=1)
        blur_score = laplacian.var().item()
        return min(1.0, blur_score * 10)

    @torch.no_grad()
    def compute_overall_score(self, frame: np.ndarray) -> Tuple[float, float, float, float, float]:
        # Convert numpy array to PIL Image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Transform and move to GPU
        frame_tensor = self.transform(frame_pil).to(self.device)
        
        # Compute scores using GPU
        with autocast():
            brightness_score = self.analyze_brightness(frame_tensor)
            color_score = self.analyze_color_diversity(frame_tensor)
            edge_score = self.detect_edges(frame_tensor)
            blur_score = self.detect_blur(frame_tensor)
            
            # Extract features using ResNet for additional quality assessment
            features = self.feature_extractor(frame_tensor.unsqueeze(0))
            feature_score = torch.sigmoid(features.mean()).item()
            
            # Weighted combination
            weights = {
                'brightness': 0.2,
                'color': 0.2,
                'edge': 0.2,
                'blur': 0.2,
                'features': 0.2
            }
            
            overall_score = (
                brightness_score * weights['brightness'] +
                color_score * weights['color'] +
                edge_score * weights['edge'] +
                blur_score * weights['blur'] +
                feature_score * weights['features']
            )
        
        return overall_score, brightness_score, color_score, edge_score, blur_score

class VideoProcessor:
    def __init__(self, input_path: str, output_dir: str, frames_per_video: int = 10, aesthetic_threshold: float = 0.75):
        self.input_path = input_path
        self.output_dir = output_dir
        self.frames_per_video = frames_per_video
        self.aesthetic_threshold = aesthetic_threshold
        self.frame_scores: List[FrameScore] = []
        self.frame_analyzer = FrameAnalyzer()
        
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
        
        video_year = YearExtractor.extract_year(os.path.basename(video_path))
        
        with tqdm(total=self.frames_per_video, desc=f"Processing {os.path.basename(video_path)}") as pbar:
            frame_count = 0
            frames_batch = []
            timestamps_batch = []
            paths_batch = []
            
            while len(frame_scores) < self.frames_per_video and frame_count < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                timestamp = frame_count / fps
                frame_filename = f"{os.path.basename(video_path)}_{frame_count:08d}.jpg"
                frame_path = os.path.join(self.base_output_dir, frame_filename)
                
                frames_batch.append(frame)
                timestamps_batch.append(timestamp)
                paths_batch.append(frame_path)
                
                if len(frames_batch) >= 8 or frame_count + interval >= total_frames:  # Process in batches of 8
                    for i, (frame, timestamp, frame_path) in enumerate(zip(frames_batch, timestamps_batch, paths_batch)):
                        overall_score, brightness_score, color_score, edge_score, blur_score = (
                            self.frame_analyzer.compute_overall_score(frame)
                        )
                        
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
                    
                    frames_batch = []
                    timestamps_batch = []
                    paths_batch = []
                
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
        
        # Process videos sequentially since we're using GPU
        results = [self.process_video(video_file) for video_file in video_files]
        
        all_scores = [score for video_scores in results for score in video_scores]
        all_scores.sort(key=lambda x: x.overall_score, reverse=True)
        
        return all_scores

    def organize_frames_by_year(self, scores: List[FrameScore]):
        years_dict = {}
        for score in scores:
            if score.year:
                if score.year not in years_dict:
                    years_dict[score.year] = []
                years_dict[score.year].append(score)
        
        for year, year_scores in years_dict.items():
            base_year_dir = os.path.join(self.base_output_dir, str(year))
            aesthetic_year_dir = os.path.join(self.aesthetic_dir, str(year))
            
            os.makedirs(base_year_dir, exist_ok=True)
            os.makedirs(aesthetic_year_dir, exist_ok=True)
            
            for score in year_scores:
                new_base_path = os.path.join(base_year_dir, os.path.basename(score.frame_path))
                shutil.move(score.frame_path, new_base_path)
                score.frame_path = new_base_path
                
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
            
            aesthetic_count = sum(1 for score in frame_scores if score.overall_score >= self.aesthetic_threshold)
            print(f"\nProcessing complete!")
            print(f"Total frames processed: {len(frame_scores)}")
            print(f"Frames meeting aesthetic threshold ({self.aesthetic_threshold:.2f}): {aesthetic_count}")
            
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
    parser = argparse.ArgumentParser(description='GPU-Accelerated Video Frame Sampler with Aesthetic Scoring and Year Organization')
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