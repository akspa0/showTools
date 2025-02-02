import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional
from tqdm import tqdm
import re
import shutil
from torchvision.models import resnet18
from torch.cuda.amp import autocast
import argparse
from pathlib import Path

@dataclass
class ImageScore:
    image_path: str
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

class ImageAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        self.feature_extractor = resnet18(pretrained=True).to(self.device)
        self.feature_extractor.eval()
        
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(self.device)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).to(self.device)
        self.sobel_x = self.sobel_x.view(1, 1, 3, 3)
        self.sobel_y = self.sobel_y.view(1, 1, 3, 3)

    @torch.no_grad()
    def analyze_brightness(self, frame_tensor: torch.Tensor) -> float:
        gray = 0.2989 * frame_tensor[0] + 0.5870 * frame_tensor[1] + 0.1140 * frame_tensor[2]
        mean_brightness = gray.mean().item()
        optimal_brightness = 0.5
        brightness_score = 1 - abs(mean_brightness - optimal_brightness) * 2
        return max(0, brightness_score)

    @torch.no_grad()
    def analyze_color_diversity(self, frame_tensor: torch.Tensor) -> float:
        hist = torch.histc(frame_tensor, bins=64, min=0, max=1)
        hist = hist / hist.sum()
        non_zero = hist[hist > 0]
        color_entropy = -(non_zero * torch.log2(non_zero)).sum().item()
        return min(1.0, color_entropy / 6.0)

    @torch.no_grad()
    def detect_edges(self, frame_tensor: torch.Tensor) -> float:
        gray = 0.2989 * frame_tensor[0] + 0.5870 * frame_tensor[1] + 0.1140 * frame_tensor[2]
        gray = gray.unsqueeze(0).unsqueeze(0)
        
        gx = nn.functional.conv2d(gray, self.sobel_x, padding=1)
        gy = nn.functional.conv2d(gray, self.sobel_y, padding=1)
        
        edge_magnitude = torch.sqrt(gx ** 2 + gy ** 2)
        edge_score = edge_magnitude.mean().item()
        return min(1.0, edge_score * 5)

    @torch.no_grad()
    def detect_blur(self, frame_tensor: torch.Tensor) -> float:
        gray = 0.2989 * frame_tensor[0] + 0.5870 * frame_tensor[1] + 0.1140 * frame_tensor[2]
        gray = gray.unsqueeze(0).unsqueeze(0)
        
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).to(self.device).view(1, 1, 3, 3)
        
        laplacian = nn.functional.conv2d(gray, laplacian_kernel, padding=1)
        blur_score = laplacian.var().item()
        return min(1.0, blur_score * 10)

    @torch.no_grad()
    def compute_overall_score(self, image_path: str) -> Tuple[float, float, float, float, float]:
        try:
            # Load and convert image
            image = Image.open(image_path).convert('RGB')
            frame_tensor = self.transform(image).to(self.device)
            
            with autocast():
                brightness_score = self.analyze_brightness(frame_tensor)
                color_score = self.analyze_color_diversity(frame_tensor)
                edge_score = self.detect_edges(frame_tensor)
                blur_score = self.detect_blur(frame_tensor)
                
                features = self.feature_extractor(frame_tensor.unsqueeze(0))
                feature_score = torch.sigmoid(features.mean()).item()
                
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
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return 0.0, 0.0, 0.0, 0.0, 0.0

class ImageProcessor:
    def __init__(self, input_dir: str, output_dir: str, aesthetic_threshold: float = 0.75):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.aesthetic_threshold = aesthetic_threshold
        self.analyzer = ImageAnalyzer()
        
        # Create output directory structure
        self.aesthetic_dir = os.path.join(output_dir, f'{aesthetic_threshold:.2f}_aesthetic')
        os.makedirs(self.aesthetic_dir, exist_ok=True)

    def process_images(self) -> List[ImageScore]:
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(Path(self.input_dir).rglob(f'*{ext}'))
            image_files.extend(Path(self.input_dir).rglob(f'*{ext.upper()}'))
        
        if not image_files:
            print("No image files found in the input directory.")
            return []
        
        scores = []
        batch_size = 32  # Process images in batches
        
        for i in tqdm(range(0, len(image_files), batch_size), desc="Processing images"):
            batch = image_files[i:i + batch_size]
            
            for image_path in batch:
                try:
                    overall_score, brightness_score, color_score, edge_score, blur_score = (
                        self.analyzer.compute_overall_score(str(image_path))
                    )
                    
                    year = YearExtractor.extract_year(image_path.name)
                    
                    scores.append(ImageScore(
                        image_path=str(image_path),
                        overall_score=overall_score,
                        brightness_score=brightness_score,
                        color_score=color_score,
                        edge_score=edge_score,
                        blur_score=blur_score,
                        year=year
                    ))
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
        
        # Sort by score
        scores.sort(key=lambda x: x.overall_score, reverse=True)
        return scores

    def organize_by_year(self, scores: List[ImageScore]):
        years_dict = {}
        for score in scores:
            if score.overall_score >= self.aesthetic_threshold:
                year = score.year if score.year else "unknown"
                if year not in years_dict:
                    years_dict[year] = []
                years_dict[year].append(score)
        
        # Create year directories and copy files
        for year, year_scores in years_dict.items():
            year_dir = os.path.join(self.aesthetic_dir, str(year))
            os.makedirs(year_dir, exist_ok=True)
            
            for score in year_scores:
                try:
                    # Create filename with score
                    base_name = os.path.basename(score.image_path)
                    new_name = f"score_{score.overall_score:.3f}_{base_name}"
                    dst_path = os.path.join(year_dir, new_name)
                    
                    # Copy file
                    shutil.copy2(score.image_path, dst_path)
                except Exception as e:
                    print(f"Error copying {score.image_path}: {str(e)}")

    def process_and_organize(self):
        print(f"Processing images in {self.input_dir}...")
        scores = self.process_images()
        
        if scores:
            print("\nOrganizing images by year and aesthetic score...")
            self.organize_by_year(scores)
            
            # Print statistics
            total_images = len(scores)
            aesthetic_images = sum(1 for s in scores if s.overall_score >= self.aesthetic_threshold)
            
            print(f"\nProcessing complete!")
            print(f"Total images processed: {total_images}")
            print(f"Images meeting aesthetic threshold ({self.aesthetic_threshold:.2f}): {aesthetic_images}")
            
            # Year-wise statistics
            years = set(score.year for score in scores if score.year is not None)
            for year in sorted(years):
                year_images = [s for s in scores if s.year == year]
                year_aesthetic = sum(1 for s in year_images if s.overall_score >= self.aesthetic_threshold)
                print(f"\nYear {year}:")
                print(f"  Total images: {len(year_images)}")
                print(f"  Aesthetic images: {year_aesthetic}")
                if year_aesthetic > 0:
                    avg_score = sum(s.overall_score for s in year_images if s.overall_score >= self.aesthetic_threshold) / year_aesthetic
                    print(f"  Average aesthetic score: {avg_score:.3f}")
        else:
            print("No images were processed.")

def main():
    parser = argparse.ArgumentParser(description='GPU-Accelerated Image Aesthetic Scorer')
    parser.add_argument('--input_dir', required=True, help='Input directory containing images')
    parser.add_argument('--output_dir', required=True, help='Output directory for organized images')
    parser.add_argument('--aesthetic_threshold', type=float, default=0.75,
                      help='Threshold for aesthetic image selection (0.0 to 1.0)')
    
    args = parser.parse_args()
    
    if not (0 <= args.aesthetic_threshold <= 1):
        parser.error("Aesthetic threshold must be between 0.0 and 1.0")
    
    processor = ImageProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        aesthetic_threshold=args.aesthetic_threshold
    )
    
    processor.process_and_organize()

if __name__ == "__main__":
    main()