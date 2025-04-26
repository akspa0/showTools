import gradio as gr
import os
import json
from whisperBite import process_audio
import tempfile
import shutil
from utils import download_audio
import logging # Add logging
import traceback # Add traceback
import re

# Configure basic logging for the app
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run_pipeline(input_file, input_folder, url, output_folder, model, num_speakers, 
                auto_speakers, enable_vocal_separation, 
                enable_word_extraction, enable_second_pass, 
                hf_token):
    """Run the audio processing pipeline based on user inputs."""
    logging.info("Starting pipeline run...")
    # Set Hugging Face token if provided
    if hf_token:
        logging.info("Setting HF_TOKEN environment variable.")
        os.environ['HF_TOKEN'] = hf_token
    else:
        logging.warning("HF_TOKEN not provided. Diarization may fail.")
        # Optionally return an error early if the token is strictly required
        # return "Error: Hugging Face Token is required for diarization.", None, ""

    if not os.path.exists(output_folder):
        logging.info(f"Creating output directory: {output_folder}")
        os.makedirs(output_folder, exist_ok=True)

    # Determine input source (Restore folder logic)
    input_path = None
    source_type = ""
    if url:
        logging.info(f"Processing URL: {url}")
        try:
            # Use a temporary directory within the output folder for downloads
            download_dir = os.path.join(output_folder, "downloads")
            os.makedirs(download_dir, exist_ok=True)
            input_path = download_audio(url, download_dir)
            source_type = "URL"
            if not input_path:
                 raise ValueError("Download failed or returned no path.")
            logging.info(f"Downloaded audio to: {input_path}")
        except Exception as download_err:
            logging.error(f"Error downloading URL {url}: {download_err}")
            return f"Error downloading URL: {str(download_err)}", None, ""
    elif input_file is not None:
        input_path = input_file.name # Use .name attribute for Gradio File component
        logging.info(f"Processing uploaded file: {input_path}")
        source_type = "File"
    elif input_folder and os.path.isdir(input_folder):
        input_path = input_folder
        logging.info(f"Processing folder: {input_path}")
        source_type = "Folder"
    elif input_folder: # Handle case where input_folder is provided but not a valid directory
        logging.error(f"Input folder path is not a valid directory: {input_folder}")
        return f"Error: Input folder path is not a valid directory: {input_folder}", None, ""
    else:
        logging.warning("No valid input provided (file, folder, or URL).")
        return "Please provide an input file, folder, or URL.", None, "" 

    if not input_path:
        logging.error("Input path could not be determined.")
        return "Error determining input path.", None, ""
        
    logging.info(f"Final input path for processing: {input_path}")

    # Run the processing pipeline
    try:
        logging.info(f"Calling process_audio with options: model={model}, num_speakers={num_speakers}, auto={auto_speakers}, separation={enable_vocal_separation}, words={enable_word_extraction}, second_pass={enable_second_pass}")
        # Pass the new arguments
        result_dir = process_audio(
            input_path=input_path,
            output_dir=output_folder,
            model_name=model,
            enable_vocal_separation=enable_vocal_separation,
            num_speakers=num_speakers,
            auto_speakers=auto_speakers,
            enable_word_extraction=enable_word_extraction, # Pass new arg
            enable_second_pass=enable_second_pass      # Pass new arg
        )

        if not result_dir or not os.path.isdir(result_dir):
            logging.error(f"process_audio did not return a valid directory path. Got: {result_dir}")
            # Restore folder-specific fallback logic (though whisperBite currently only processes one file)
            if source_type == "Folder":
                 # This fallback might be inaccurate if whisperBite changes, but restores previous behavior
                 result_dir = output_folder 
                 logging.warning(f"process_audio returned invalid path for folder input. Assuming results are in main output folder: {result_dir}")
            else:
                 return "Processing finished, but the result directory was not found.", None, ""

        logging.info(f"Processing finished. Looking for results in: {result_dir}")

        # Find the results zip file and transcript within the specific result directory
        result_zip_file = None
        transcript = ""
        master_transcript_path = os.path.join(result_dir, "master_transcript.txt")
        second_pass_transcript_path = os.path.join(result_dir, "2nd_pass", "master_transcript.txt")

        # Prefer second pass transcript if it exists
        transcript_path_to_read = None
        if enable_second_pass and os.path.exists(second_pass_transcript_path):
            transcript_path_to_read = second_pass_transcript_path
            logging.info(f"Using second pass transcript: {transcript_path_to_read}")
        elif os.path.exists(master_transcript_path):
            transcript_path_to_read = master_transcript_path
            logging.info(f"Using first pass transcript: {transcript_path_to_read}")
        else:
            logging.warning(f"Master transcript not found in {result_dir}")

        if transcript_path_to_read:
            try:
                with open(transcript_path_to_read, 'r', encoding='utf-8') as f:
                    transcript = f.read()
            except Exception as read_err:
                logging.error(f"Error reading transcript file {transcript_path_to_read}: {read_err}")
                transcript = f"[Error reading transcript: {read_err}]"

        # --- Construct the expected zip file path ---
        # The zip file is saved in the parent directory of result_dir
        result_zip_file = None # Initialize to None
        try:
            # Extract the base name used for the zip from the result_dir name
            # Assumes result_dir is like '.../output/INPUT_BASENAME_TIMESTAMP'
            result_dir_name = os.path.basename(result_dir) # e.g., "INPUT_BASENAME_TIMESTAMP"
            # Find the last timestamp pattern (e.g., _YYYYMMDD_HHMMSS)
            timestamp_pattern = r"_(\d{8}_\d{6})$"
            match = re.search(timestamp_pattern, result_dir_name)
            if match:
                original_input_basename = result_dir_name[:match.start()] # Get part before timestamp
                
                parent_dir = os.path.dirname(result_dir) # e.g., "./whisper_output"
                # Reconstruct the zip filename pattern used in utils.zip_results
                # zip_filename = os.path.join(parent_dir, f"{base_name}_results_{os.path.basename(output_dir)}.zip")
                expected_zip_filename = f"{original_input_basename}_results_{result_dir_name}.zip"
                expected_zip_path = os.path.join(parent_dir, expected_zip_filename)

                if os.path.exists(expected_zip_path):
                    result_zip_file = expected_zip_path
                    logging.info(f"Found result zip file: {result_zip_file}")
                else:
                    logging.warning(f"Expected zip file not found at: {expected_zip_path}")
            else:
                logging.warning(f"Could not extract base name and timestamp from result directory: {result_dir_name}. Cannot reliably locate zip file.")
        
        except Exception as zip_find_err:
            logging.error(f"Error constructing or finding zip file path: {zip_find_err}")
        # --- End zip file path construction ---

        results_message = f"Processing complete! Results saved to {result_dir}"
        
        # Copy result file to temp dir for Gradio access
        final_result_path_for_gradio = None
        if result_zip_file:
            results_message += f"\nZip file created: {os.path.basename(result_zip_file)}"
            try:
                # Create a unique temp dir for this request
                temp_dir = tempfile.mkdtemp()
                temp_file_path = os.path.join(temp_dir, os.path.basename(result_zip_file))
                shutil.copy2(result_zip_file, temp_file_path)
                final_result_path_for_gradio = temp_file_path
                logging.info(f"Copied zip to temp location for download: {final_result_path_for_gradio}")
                 # Consider cleaning up older temp dirs if they accumulate
            except Exception as copy_err:
                logging.error(f"Error copying result zip to temp dir: {copy_err}")
                results_message += f"\nError preparing zip for download: {copy_err}"
                final_result_path_for_gradio = None # Ensure it's None if copy fails
        else:
             logging.warning("No result zip file found.")
             results_message += "\nResult zip file not found."
            
        return results_message, final_result_path_for_gradio, transcript
    except Exception as e:
        logging.error(f"An error occurred during pipeline execution: {e}")
        logging.error(traceback.format_exc())
        return f"An error occurred: {str(e)}\n\n{traceback.format_exc()}", None, ""

# Gradio interface
def build_interface():
    with gr.Blocks(title="WhisperBite - Audio Processing Tool") as demo:
        gr.Markdown("# 🎙️ WhisperBite")
        gr.Markdown("""
        This tool processes audio files by:
        1. Normalizing audio levels
        2. Separating speakers (diarization)
        3. Transcribing each speaker's audio
        4. Creating individual soundbites with transcripts
        """)

        with gr.Tabs():
            with gr.TabItem("Input"):
                with gr.Row():
                    with gr.Column():
                        # Add common video extensions
                        input_file = gr.File(
                            label="Input Audio or Video File", 
                            file_types=[
                                ".wav", ".mp3", ".m4a", ".ogg", ".flac", # Audio
                                ".mp4", ".mov", ".avi", ".mkv", ".webm" # Video
                            ]
                        )
                    with gr.Column():
                        input_folder = gr.Textbox(label="Input Folder Path", placeholder="Path to folder containing audio/video files", info="Processes the newest audio/video file in the folder.")
                    with gr.Column():
                        url = gr.Textbox(label="Audio URL", placeholder="YouTube or direct audio URL")
                
                output_folder = gr.Textbox(
                    label="Output Folder", 
                    placeholder="Path to save results",
                    value="./whisper_output"
                )
                
            with gr.TabItem("Processing Options"):
                with gr.Row():
                    model = gr.Dropdown(
                        label="Whisper Model", 
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"],
                        value="turbo", # Default set to turbo
                        info="Larger models are more accurate but slower"
                    )
                    
                    num_speakers = gr.Slider(
                        label="Number of Speakers", 
                        minimum=1, 
                        maximum=10, 
                        step=1, 
                        value=2,
                        info="Set expected number of speakers"
                    )
                
                with gr.Row():
                    auto_speakers = gr.Checkbox(
                        label="Auto-detect Speaker Count", 
                        value=True,
                        info="Automatically determine optimal speaker count"
                    )
                    
                    enable_vocal_separation = gr.Checkbox(
                        label="Enable Vocal Separation", 
                        value=False,
                        info="Isolate voices from background noise/music (requires Demucs)"
                    )
                
                # Add new options here
                with gr.Row():
                    enable_word_extraction = gr.Checkbox(
                        label="Enable Word Audio Extraction", 
                        value=False,
                        info="Extract individual audio files for each word (can create many files)"
                    )
                    enable_second_pass = gr.Checkbox(
                        label="Enable Second Pass Refinement", 
                        value=False, 
                        info="Re-analyze initial segments to improve speaker separation (experimental)"
                    )

                hf_token = gr.Textbox(
                    label="Hugging Face Token (Required)", 
                    placeholder="Set your Hugging Face token for pyannote.audio",
                    type="password",
                    info="Get token at huggingface.co/settings/tokens"
                )
        
        submit_button = gr.Button("Process Audio", variant="primary")
        
        with gr.Row():
            output_message = gr.Textbox(label="Status", interactive=False, lines=3) # Increased lines for better messages
            result_file = gr.File(label="Download Results")
            transcript_preview = gr.TextArea(label="Transcript Preview", interactive=False, lines=10)

        submit_button.click(
            fn=run_pipeline,
            inputs=[
                input_file, input_folder, url, output_folder, model, 
                num_speakers, auto_speakers, enable_vocal_separation,
                enable_word_extraction, enable_second_pass,
                hf_token
            ],
            outputs=[output_message, result_file, transcript_preview]
        )
        
        # Add sample URLs only since file examples cause issues
        gr.Markdown("""
        ### Sample URLs to try:
        - Short video: https://www.youtube.com/watch?v=jNQXAC9IVRw
        - Interview example: https://www.youtube.com/watch?v=8S0FDjFBj8o
        """)
        
        # Examples with file paths don't work well with Gradio
        # Use sample configurations instead
        with gr.Accordion("Sample Configurations", open=False):
            gr.Markdown("""
            **Single Speaker Podcast:**
            - Model: base
            - Speakers: 1
            - Auto-detect: Disabled
            - Vocal Separation: Enabled
            
            **Interview Setup:**
            - Model: small
            - Speakers: 2
            - Auto-detect: Enabled
            - Vocal Separation: Enabled
            
            **Group Discussion:**
            - Model: medium
            - Speakers: 4
            - Auto-detect: Enabled
            - Vocal Separation: Disabled
            """)

    return demo

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="WhisperBite - Audio Processing Tool")
    parser.add_argument("--public", action="store_true", help="Make the Gradio interface publicly accessible")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the Gradio interface on")
    
    args = parser.parse_args()
    
    interface = build_interface()
    interface.launch(share=args.public, server_port=args.port) # Removed allowed_paths
