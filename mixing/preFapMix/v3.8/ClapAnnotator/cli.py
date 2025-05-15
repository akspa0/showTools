#!/usr/bin/env python
import argparse
import logging
import sys
import json
from pathlib import Path
import traceback
from typing import List, Dict, Optional, Union
import shutil

# Set up basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# Import project modules and settings with error handling
try:
    from utils.logging_config import setup_logging
    setup_logging()
    
    from config import settings
    from utils.file_utils import generate_output_path, cleanup_directory, ensure_dir
    from utils.audio_utils import resample_audio_ffmpeg
    from utils.preset_utils import load_clap_prompt_presets
    from audio_separation.separator import AudioSeparatorWrapper
    from clap_annotation.annotator import CLAPAnnotatorWrapper
    
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    log.error(f"Failed to import required modules: {e}")
    log.error("Please ensure all dependencies are installed: pip install -r requirements.txt")
    IMPORTS_SUCCESSFUL = False
except Exception as e:
    log.error(f"Unexpected error during imports: {e}")
    log.error(traceback.format_exc())
    IMPORTS_SUCCESSFUL = False

def process_audio_file(
    input_file: Path,
    separator_model: str,
    prompts: List[str],
    confidence_threshold: float,
    output_dir: Optional[Path] = None,
    cleanup_temp: bool = True,
    keep_audio: bool = True,
    chunk_duration: int = settings.CLAP_CHUNK_DURATION_S
) -> Dict:
    """
    Process a single audio file through the CLAP Annotator pipeline.
    
    Args:
        input_file: Path to the input audio file
        separator_model: Name of the audio separator model to use
        prompts: List of text prompts for CLAP annotation
        confidence_threshold: Confidence threshold for CLAP detections
        output_dir: Optional custom output directory (uses default if None)
        cleanup_temp: Whether to clean up temporary files after processing
        keep_audio: Whether to copy audio files to the output directory
        chunk_duration: Duration in seconds for each CLAP analysis chunk
        
    Returns:
        Dictionary containing the processing results
    """
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if not prompts:
        raise ValueError("No prompts provided. Please specify at least one prompt.")
        
    log.info(f"Processing file: {input_file}")
    log.info(f"Using separator model: {separator_model}")
    log.info(f"Using {len(prompts)} prompts with confidence threshold: {confidence_threshold}")
    log.info(f"CLAP chunk duration: {chunk_duration} seconds")
    
    # Create output directory
    if output_dir is None:
        output_dir = generate_output_path(settings.BASE_OUTPUT_DIR, str(input_file))
    else:
        ensure_dir(output_dir)
    
    log.info(f"Output will be saved to: {output_dir}")
    
    # Ensure temp directory exists
    ensure_dir(settings.TEMP_OUTPUT_DIR)
    
    try:
        # Step 1: Separate audio
        log.info("Step 1: Separating audio...")
        model_config = settings.AUDIO_SEPARATOR_AVAILABLE_MODELS.get(separator_model)
        if not model_config:
            raise ValueError(f"Unknown separator model: {separator_model}")
        
        separator = AudioSeparatorWrapper(
            model_name=model_config["model_name"],
            output_dir=settings.TEMP_OUTPUT_DIR,
            **model_config.get("params", {})
        )
        
        separated_stems = separator.separate(input_file)
        if not separated_stems:
            raise RuntimeError("Audio separation failed to produce any stems")
        
        log.info(f"Separated stems: {', '.join(separated_stems.keys())}")
        
        # Step 2: Resample audio for CLAP
        log.info("Step 2: Resampling audio for CLAP...")
        resampled_stems = {}
        for stem_name, stem_path in separated_stems.items():
            resampled_path = settings.TEMP_OUTPUT_DIR / f"{stem_path.stem}_resampled_{settings.CLAP_EXPECTED_SR}Hz.wav"
            resample_audio_ffmpeg(
                input_path=stem_path,
                output_path=resampled_path,
                target_sr=settings.CLAP_EXPECTED_SR
            )
            resampled_stems[stem_name] = resampled_path
            log.info(f"Resampled {stem_name} to {resampled_path}")
        
        # Step 3: Annotate with CLAP
        log.info("Step 3: Annotating with CLAP...")
        clap_annotator = CLAPAnnotatorWrapper(
            model_name=settings.CLAP_MODEL_NAME,
            chunk_duration_s=chunk_duration,
            expected_sr=settings.CLAP_EXPECTED_SR
        )
        
        results = {
            "input_file": str(input_file),
            "separator_model": separator_model,
            "clap_model": settings.CLAP_MODEL_NAME,
            "confidence_threshold": confidence_threshold,
            "prompts": prompts,
            "chunk_duration_s": chunk_duration,
            "stems": {},
            "preserved_audio_files": {}
        }
        
        for stem_name, resampled_path in resampled_stems.items():
            log.info(f"Annotating {stem_name}...")
            try:
                stem_results = clap_annotator.annotate(
                    audio_path=resampled_path,
                    text_prompts=prompts,
                    confidence_threshold=confidence_threshold
                )
                results["stems"][stem_name] = stem_results
                log.info(f"Found {len(stem_results['detections'])} detections for {stem_name}")
            except Exception as e:
                log.error(f"Error annotating {stem_name}: {e}")
                results["stems"][stem_name] = {"error": str(e)}
        
        # Step 4: Copy audio files to output directory if keeping them
        preserved_files = {}
        if keep_audio:
            log.info("Step 4: Preserving audio files...")
            for stem_name, resampled_path in resampled_stems.items():
                # Copy the resampled file to the output directory
                dest_path = output_dir / resampled_path.name
                try:
                    shutil.copy2(resampled_path, dest_path)
                    log.info(f"Copied {stem_name} audio to {dest_path}")
                    
                    # Store the relative path for the preserved file
                    rel_path = dest_path.relative_to(settings.PROJECT_ROOT)
                    preserved_files[stem_name] = str(rel_path)
                    
                    # Update the file path in the results to point to the preserved copy
                    if stem_name in results["stems"]:
                        results["stems"][stem_name]["file"] = str(rel_path)
                except Exception as e:
                    log.error(f"Failed to copy {stem_name} audio: {e}")
        else:
            log.info("Step 4: Skipping audio preservation (--keep-audio not specified)")
        
        # Add preserved files to results
        results["preserved_audio_files"] = preserved_files
        
        # Step 5: Save results
        log.info("Step 5: Saving results...")
        results_file = output_dir / f"{input_file.stem}_clap_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        log.info(f"Results saved to {results_file}")
        
        return {
            "success": True,
            "output_file": str(results_file),
            "results": results
        }
    
    except Exception as e:
        log.exception(f"Error processing {input_file}: {e}")
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
    finally:
        # Cleanup temporary files if requested
        if cleanup_temp:
            log.info("Cleaning up temporary files...")
            cleanup_directory(settings.TEMP_OUTPUT_DIR)

def check_environment():
    """Check if the environment is properly configured."""
    issues = []
    
    # Check Python version
    import platform
    python_version = platform.python_version()
    if not python_version.startswith('3.10'):
        issues.append(f"Python version is {python_version}, but 3.10+ is recommended")
    
    # Check for ffmpeg
    import shutil
    if not shutil.which('ffmpeg'):
        issues.append("ffmpeg not found in PATH. Please install ffmpeg and ensure it's in your PATH")
    
    # Check for critical packages
    try:
        import numpy
        log.info(f"NumPy version: {numpy.__version__}")
    except ImportError:
        issues.append("NumPy not installed. Please run: pip install numpy>=1.23.5,<1.25.0")
    
    try:
        import scipy
        log.info(f"SciPy version: {scipy.__version__}")
    except ImportError:
        issues.append("SciPy not installed. Please run: pip install scipy>=1.10.0,<1.11.0")
    
    try:
        import torch
        log.info(f"PyTorch version: {torch.__version__}")
        log.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log.info(f"CUDA version: {torch.version.cuda}")
    except ImportError:
        issues.append("PyTorch not installed. Please run: pip install torch>=2.0.0")
    
    try:
        import transformers
        log.info(f"Transformers version: {transformers.__version__}")
    except ImportError:
        issues.append("Transformers not installed. Please run: pip install transformers[torch]>=4.30.0")
    
    # Check for .env file
    if not Path('.env').is_file():
        issues.append("No .env file found. Please create one with your HF_TOKEN")
    
    return issues

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="CLAP Annotator CLI - Process audio files with CLAP annotation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add a check-environment command
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check if the environment is properly configured"
    )
    
    # Input options
    parser.add_argument(
        "input", 
        type=str, 
        nargs="?",  # Make input optional when using --check-env
        help="Path to input audio file or directory containing audio files"
    )
    parser.add_argument(
        "--batch", 
        action="store_true", 
        help="Process all audio files in the input directory"
    )
    
    # If imports succeeded, add the remaining options
    if IMPORTS_SUCCESSFUL:
        # Load presets once
        all_presets = load_clap_prompt_presets()
        preset_choices = list(all_presets.keys())
        separator_model_choices = list(settings.AUDIO_SEPARATOR_AVAILABLE_MODELS.keys())
        
        # Model options
        parser.add_argument(
            "--separator-model", 
            type=str, 
            choices=separator_model_choices,
            default=settings.DEFAULT_AUDIO_SEPARATOR_MODEL,
            help="Audio separator model to use"
        )
        
        # CLAP options
        prompt_group = parser.add_mutually_exclusive_group()
        prompt_group.add_argument(
            "--prompts", 
            type=str, 
            help="Comma-separated list of CLAP prompts"
        )
        prompt_group.add_argument(
            "--preset", 
            type=str, 
            choices=preset_choices if preset_choices else None,
            help="Name of CLAP prompt preset to use"
        )
        prompt_group.add_argument(
            "--prompt-file", 
            type=str, 
            help="Path to a text file containing prompts (one per line)"
        )
        
        parser.add_argument(
            "--confidence", 
            type=float, 
            default=settings.DEFAULT_CLAP_CONFIDENCE_THRESHOLD,
            help="Confidence threshold for CLAP detections"
        )
        
        parser.add_argument(
            "--chunk-duration",
            type=int,
            default=settings.CLAP_CHUNK_DURATION_S,
            help=f"Duration in seconds for each CLAP analysis chunk"
        )
        
        # Output options
        parser.add_argument(
            "--output-dir", 
            type=str, 
            help="Custom output directory (default: auto-generated based on input file)"
        )
        parser.add_argument(
            "--keep-temp", 
            action="store_true", 
            help="Keep temporary files after processing"
        )
        parser.add_argument(
            "--keep-audio", 
            action="store_true", 
            default=True,
            help="Save separated audio files to output directory"
        )
        parser.add_argument(
            "--no-keep-audio", 
            action="store_false", 
            dest="keep_audio",
            help="Don't save separated audio files, only keep JSON results"
        )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if environment check was requested
    if args.check_env:
        log.info("Checking environment...")
        issues = check_environment()
        if issues:
            log.warning("Environment check found issues:")
            for i, issue in enumerate(issues, 1):
                log.warning(f"  {i}. {issue}")
            log.info("Please fix these issues and try again.")
        else:
            log.info("Environment check passed! All dependencies appear to be installed correctly.")
        return
    
    # If imports failed, exit early
    if not IMPORTS_SUCCESSFUL:
        log.error("Critical imports failed. Please fix the issues above and try again.")
        log.info("Run 'python cli.py --check-env' to check your environment.")
        sys.exit(1)
    
    # Check if input is provided
    if args.input is None:
        parser.print_help()
        sys.exit(1)
    
    # Get prompts from the appropriate source
    prompts = []
    if args.prompts:
        prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    elif args.preset:
        if args.preset in all_presets:
            prompts = all_presets[args.preset]
        else:
            log.error(f"Preset '{args.preset}' not found")
            sys.exit(1)
    elif args.prompt_file:
        try:
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
        except Exception as e:
            log.error(f"Error reading prompt file: {e}")
            sys.exit(1)
    
    if not prompts:
        log.error("No prompts provided. Please use --prompts, --preset, or --prompt-file")
        sys.exit(1)
    
    # Set up output directory if specified
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Process input
    input_path = Path(args.input)
    
    if args.batch and input_path.is_dir():
        # Batch processing
        log.info(f"Batch processing all audio files in: {input_path}")
        audio_files = []
        for ext in [".mp3", ".wav", ".flac", ".m4a", ".ogg"]:
            audio_files.extend(input_path.glob(f"*{ext}"))
        
        if not audio_files:
            log.error(f"No audio files found in {input_path}")
            sys.exit(1)
        
        log.info(f"Found {len(audio_files)} audio files to process")
        
        results = []
        for audio_file in audio_files:
            log.info(f"Processing {audio_file.name}...")
            file_output_dir = output_dir / audio_file.stem if output_dir else None
            result = process_audio_file(
                input_file=audio_file,
                separator_model=args.separator_model,
                prompts=prompts,
                confidence_threshold=args.confidence,
                output_dir=file_output_dir,
                cleanup_temp=not args.keep_temp,
                keep_audio=args.keep_audio,
                chunk_duration=args.chunk_duration
            )
            results.append({"file": str(audio_file), "result": result})
        
        # Save batch results summary
        if output_dir:
            batch_results_file = output_dir / "batch_results_summary.json"
            with open(batch_results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            log.info(f"Batch results summary saved to {batch_results_file}")
        
        # Print summary
        success_count = sum(1 for r in results if r["result"].get("success", False))
        log.info(f"Batch processing complete: {success_count}/{len(results)} files processed successfully")
        
    elif input_path.is_file():
        # Single file processing
        result = process_audio_file(
            input_file=input_path,
            separator_model=args.separator_model,
            prompts=prompts,
            confidence_threshold=args.confidence,
            output_dir=output_dir,
            cleanup_temp=not args.keep_temp,
            keep_audio=args.keep_audio,
            chunk_duration=args.chunk_duration
        )
        
        if result.get("success", False):
            log.info("Processing completed successfully")
            log.info(f"Results saved to {result['output_file']}")
        else:
            log.error(f"Processing failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    else:
        log.error(f"Input path does not exist or is not a file/directory: {input_path}")
        sys.exit(1)

if __name__ == "__main__":
    main() 