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

[Previous ImageScore, YearExtractor, and ImageAnalyzer classes remain the same...]

class ImageProcessor:
    def __init__(self, input_dir: str, output_dir: str, aesthetic_threshold: float = 0.75,
                 sequential_naming: bool = False, flat_dir: bool = False):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.aesthetic_threshold = aesthetic_threshold
        self.sequential_naming = sequential_naming
        self.flat_dir = flat_dir
        self.analyzer = ImageAnalyzer()
        
        # Create output directory structure
        self.aesthetic_dir = os.path.join(output_dir, f'{aesthetic_threshold:.2f}_aesthetic')
        if flat_dir:
            # Use a single directory for flat organization
            self.aesthetic_dir = os.path.join(output_dir, f'{aesthetic_threshold:.2f}_aesthetic_flat')
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
        batch_size = 32
        
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
        # Filter scores that meet the threshold
        aesthetic_scores = [s for s in scores if s.overall_score >= self.aesthetic_threshold]
        
        if not aesthetic_scores:
            print(f"No images met the aesthetic threshold of {self.aesthetic_threshold}")
            return
        
        if self.flat_dir:
            # Organize in a flat directory with sequential naming
            for i, score in enumerate(aesthetic_scores, 1):
                try:
                    ext = os.path.splitext(score.image_path)[1]
                    if self.sequential_naming:
                        new_name = f"{i:06d}{ext}"
                    else:
                        new_name = f"score_{score.overall_score:.3f}_{i:06d}{ext}"
                    
                    dst_path = os.path.join(self.aesthetic_dir, new_name)
                    shutil.copy2(score.image_path, dst_path)
                except Exception as e:
                    print(f"Error copying {score.image_path}: {str(e)}")
        else:
            # Organize by year with optional sequential naming
            years_dict = {}
            for score in aesthetic_scores:
                year = score.year if score.year else "unknown"
                if year not in years_dict:
                    years_dict[year] = []
                years_dict[year].append(score)
            
            # Counter for sequential naming across all years
            global_counter = 1
            
            for year, year_scores in years_dict.items():
                year_dir = os.path.join(self.aesthetic_dir, str(year))
                os.makedirs(year_dir, exist_ok=True)
                
                for score in year_scores:
                    try:
                        ext = os.path.splitext(score.image_path)[1]
                        if self.sequential_naming:
                            new_name = f"{global_counter:06d}{ext}"
                            global_counter += 1
                        else:
                            new_name = f"score_{score.overall_score:.3f}_{os.path.basename(score.image_path)}"
                        
                        dst_path = os.path.join(year_dir, new_name)
                        shutil.copy2(score.image_path, dst_path)
                    except Exception as e:
                        print(f"Error copying {score.image_path}: {str(e)}")

    def process_and_organize(self):
        print(f"Processing images in {self.input_dir}...")
        scores = self.process_images()
        
        if scores:
            print("\nOrganizing images...")
            self.organize_by_year(scores)
            
            # Print statistics
            total_images = len(scores)
            aesthetic_images = sum(1 for s in scores if s.overall_score >= self.aesthetic_threshold)
            
            print(f"\nProcessing complete!")
            print(f"Total images processed: {total_images}")
            print(f"Images meeting aesthetic threshold ({self.aesthetic_threshold:.2f}): {aesthetic_images}")
            
            if not self.flat_dir:
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
    parser.add_argument('--sequential_naming', action='store_true',
                      help='Use sequential numbering for output files (e.g., 000001.jpg)')
    parser.add_argument('--flat_dir', action='store_true',
                      help='Store all outputs in a single flat directory instead of organizing by year')
    
    args = parser.parse_args()
    
    if not (0 <= args.aesthetic_threshold <= 1):
        parser.error("Aesthetic threshold must be between 0.0 and 1.0")
    
    processor = ImageProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        aesthetic_threshold=args.aesthetic_threshold,
        sequential_naming=args.sequential_naming,
        flat_dir=args.flat_dir
    )
    
    processor.process_and_organize()

if __name__ == "__main__":
    main()