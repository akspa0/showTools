import os
import logging
import subprocess
import tempfile
import shutil

def separate_vocals_with_demucs(input_audio, output_dir, model="htdemucs"):
    """
    Separate vocals from an audio file using Demucs.
    
    Args:
        input_audio (str): Path to input audio file
        output_dir (str): Directory to save output
        model (str): Demucs model to use (htdemucs, htdemucs_ft, mdx, mdx_extra)
        
    Returns:
        str: Path to the extracted vocals file
    """
    demucs_output_dir = os.path.join(output_dir, "demucs")
    os.makedirs(demucs_output_dir, exist_ok=True)

    try:
        logging.info(f"Running Demucs ({model}) on {input_audio}")
        
        # First verify demucs is installed
        try:
            # Try direct command
            version_proc = subprocess.run(["demucs", "--version"], capture_output=True, text=True)
            if version_proc.returncode == 0:
                demucs_path = "demucs"
                logging.info(f"Found demucs: {version_proc.stdout.strip()}")
            else:
                # Try to locate demucs using pip
                pip_proc = subprocess.run(["pip", "show", "demucs"], capture_output=True, text=True)
                if "Location:" in pip_proc.stdout:
                    # Extract package location
                    package_loc = None
                    for line in pip_proc.stdout.split('\n'):
                        if line.startswith("Location:"):
                            package_loc = line.split("Location:")[1].strip()
                            break
                    
                    if package_loc:
                        # Look for executable in common relative paths
                        possible_paths = [
                            os.path.join(os.path.dirname(package_loc), "bin", "demucs"),
                            os.path.join(package_loc, "..", "..", "bin", "demucs")
                        ]
                        
                        for path in possible_paths:
                            if os.path.exists(path) and os.access(path, os.X_OK):
                                demucs_path = path
                                logging.info(f"Found demucs at {path}")
                                break
                        else:
                            raise FileNotFoundError("Demucs executable not found")
                else:
                    raise FileNotFoundError("Demucs package information not found")
        except Exception as e:
            logging.error(f"Demucs check failed: {e}")
            logging.error("Demucs is not installed or not in PATH. Install with 'pip install demucs'")
            raise RuntimeError("Demucs not found in PATH")
        
        # Verify the model is valid
        valid_models = ["htdemucs", "htdemucs_ft", "mdx", "mdx_extra"]
        if model not in valid_models:
            logging.warning(f"Model {model} not in {valid_models}. Falling back to htdemucs.")
            model = "htdemucs"
            
        # Use a temporary directory to avoid permission issues and file clutter
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run demucs command
            cmd = [
                demucs_path,
                "--two-stems", "vocals",
                "--out", temp_dir,
                "--model", model,
                input_audio
            ]
            
            logging.info(f"Running demucs command: {' '.join(cmd)}")
            process = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            # Log output for debugging
            logging.debug(f"Demucs stdout: {process.stdout}")
            if process.stderr:
                logging.warning(f"Demucs stderr: {process.stderr}")
            
            # Find the vocals output file 
            # The structure is typically: {temp_dir}/{model}/{input_basename}/vocals.wav
            model_dir = os.path.join(temp_dir, model)
            input_base_name = os.path.splitext(os.path.basename(input_audio))[0]
            vocals_dir = os.path.join(model_dir, input_base_name)
            vocals_file = os.path.join(vocals_dir, "vocals.wav")
            
            if not os.path.exists(vocals_file):
                # Try to find any vocals file (in case the model name creates a different structure)
                logging.warning(f"Expected vocals file not found at {vocals_file}, searching for alternatives")
                
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file == "vocals.wav":
                            vocals_file = os.path.join(root, file)
                            break
                
                if not os.path.exists(vocals_file):
                    logging.error(f"Could not find vocals output anywhere in {temp_dir}")
                    raise FileNotFoundError(f"No vocals output found from Demucs")
            
            # Copy the vocals file to our output directory
            final_vocals_file = os.path.join(demucs_output_dir, f"{input_base_name}_vocals.wav")
            shutil.copy2(vocals_file, final_vocals_file)
            
            # Optionally copy the accompaniment (no-vocals) too if it exists
            no_vocals_file = os.path.join(vocals_dir, "no_vocals.wav")
            if os.path.exists(no_vocals_file):
                final_no_vocals_file = os.path.join(demucs_output_dir, f"{input_base_name}_no_vocals.wav")
                shutil.copy2(no_vocals_file, final_no_vocals_file)
                
            # Also copy any other interesting stems we find
            for stem in ["drums", "bass", "other"]:
                stem_file = os.path.join(vocals_dir, f"{stem}.wav")
                if os.path.exists(stem_file):
                    final_stem_file = os.path.join(demucs_output_dir, f"{input_base_name}_{stem}.wav")
                    shutil.copy2(stem_file, final_stem_file)
        
        logging.info(f"Vocal separation completed successfully. File saved to {final_vocals_file}")
        return final_vocals_file
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Demucs process error: {e}")
        logging.error(f"Demucs stderr: {e.stderr}")
        raise
    except Exception as e:
        logging.error(f"Error during vocal separation: {e}")
        raise

def enhance_vocals(vocals_file, output_dir):
    """
    Enhance extracted vocals with noise reduction and EQ.
    Requires ffmpeg with the afftdn filter.
    
    Args:
        vocals_file (str): Path to vocal file
        output_dir (str): Directory to save output
        
    Returns:
        str: Path to enhanced vocal file
    """
    try:
        base_name = os.path.splitext(os.path.basename(vocals_file))[0]
        enhanced_file = os.path.join(output_dir, f"{base_name}_enhanced.wav")
        
        # Apply mild noise reduction and EQ to enhance vocals
        cmd = [
            "ffmpeg", "-y",
            "-i", vocals_file,
            "-af", "afftdn=nf=-20,equalizer=f=200:t=h:width=200:g=-3,equalizer=f=3000:t=h:width=1000:g=3,compand=0.3|0.3:1|1:-90/-60|-60/-40|-40/-30|-20/-20:6:0:-90:0.2",
            "-ar", "44100",
            enhanced_file
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        logging.info(f"Enhanced vocals saved to {enhanced_file}")
        return enhanced_file
    except Exception as e:
        logging.warning(f"Error enhancing vocals: {e}. Using original vocals.")
        return vocals_file
