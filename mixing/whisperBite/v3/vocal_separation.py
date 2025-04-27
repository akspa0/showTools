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
        tuple(str, str): Paths to the extracted (vocals_file, no_vocals_file).
                         no_vocals_file will be None if not found/created.
    """
    demucs_output_dir = os.path.join(output_dir, "demucs")
    os.makedirs(demucs_output_dir, exist_ok=True)
    final_vocals_file = None
    final_no_vocals_file = None

    try:
        logging.info(f"Running Demucs ({model}) on {input_audio}")
        
        # --- Simplified Demucs Path Handling ---
        # Assume 'demucs' is in PATH if installed in the environment
        demucs_path = "demucs"
        # --- End Simplified Handling ---
        
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
                "-n", model,
                input_audio
            ]
            
            logging.info(f"[Demucs Command] Running: {' '.join(cmd)}")
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
            expected_vocals_file = os.path.join(vocals_dir, "vocals.wav")
            logging.info(f"[Demucs Output] Expecting vocals file at: {expected_vocals_file}")
            
            vocals_file = None
            if os.path.exists(expected_vocals_file):
                vocals_file = expected_vocals_file
            else:
                # Try to find any vocals file (in case the model name creates a different structure)
                logging.warning(f"Expected vocals file not found at {expected_vocals_file}, searching for alternatives in {temp_dir}")
                
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file == "vocals.wav":
                            vocals_file = os.path.join(root, file)
                            break
                
                if not vocals_file:
                    logging.error(f"Could not find vocals output anywhere in {temp_dir}")
                    raise FileNotFoundError(f"No vocals output found from Demucs")
                else:
                    logging.info(f"[Demucs Output] Found vocals file at: {vocals_file}")
            
            # Copy the vocals file to our output directory
            final_vocals_file = os.path.join(demucs_output_dir, f"{input_base_name}_vocals.wav")
            logging.info(f"[Demucs Output] Copying {vocals_file} to {final_vocals_file}")
            shutil.copy2(vocals_file, final_vocals_file)
            
            # Optionally copy the accompaniment (no-vocals) too if it exists
            no_vocals_file = os.path.join(vocals_dir, "no_vocals.wav")
            if os.path.exists(no_vocals_file):
                final_no_vocals_file = os.path.join(demucs_output_dir, f"{input_base_name}_no_vocals.wav")
                logging.info(f"[Demucs Output] Copying {no_vocals_file} to {final_no_vocals_file}")
                shutil.copy2(no_vocals_file, final_no_vocals_file)
                
            # Also copy any other interesting stems we find
            for stem in ["drums", "bass", "other"]:
                stem_file = os.path.join(vocals_dir, f"{stem}.wav")
                if os.path.exists(stem_file):
                    final_stem_file = os.path.join(demucs_output_dir, f"{input_base_name}_{stem}.wav")
                    shutil.copy2(stem_file, final_stem_file)
        
        logging.info(f"[Demucs Success] Vocal separation completed successfully. Vocals: {final_vocals_file}, No Vocals: {final_no_vocals_file}")
        return final_vocals_file, final_no_vocals_file
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Demucs process error: {e}")
        logging.error(f"Demucs command: {' '.join(e.cmd)}") # Log the command that failed
        logging.error(f"Demucs stderr: {e.stderr}")
        return None, None
    except FileNotFoundError:
        # Catch if 'demucs' command is truly not found in PATH
        logging.error("Demucs command not found. Please ensure demucs is installed in the active environment and the environment's scripts/bin directory is in your system PATH.")
        logging.error("Install with 'pip install demucs'")
        return None, None
    except Exception as e:
        logging.error(f"Error during vocal separation: {e}")
        return None, None

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
