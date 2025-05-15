import logging
import gradio as gr
import json
from pathlib import Path
import traceback
import os
import sys
import time
import shutil # Added for file copying

# Add project root to sys.path to allow importing modules from the root directory
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.append(str(project_root))

# Configure logging first (assuming it needs to be called explicitly)
from utils import logging_config
logging_config.setup_logging() # Call the setup function

# Import project modules and settings
from config import settings
from utils import file_utils, audio_utils, preset_utils
from audio_separation.separator import AudioSeparatorWrapper
from clap_annotation.annotator import CLAPAnnotatorWrapper

log = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_STATUS = "Idle. Upload an audio file and configure options."

# --- Global Variables / State (Minimize use of globals if possible) ---
# Load presets once on startup
ALL_CLAP_PRESETS = preset_utils.load_clap_prompt_presets()
PRESET_CHOICES = list(ALL_CLAP_PRESETS.keys())
SEPARATOR_MODEL_CHOICES = list(settings.AUDIO_SEPARATOR_AVAILABLE_MODELS.keys())

# --- Helper Functions ---

def _get_prompts_from_ui(preset_selection, custom_prompts_text):
    """Gets the list of prompts from either preset or custom text input."""
    if preset_selection and preset_selection in ALL_CLAP_PRESETS:
        log.info(f"Using prompts from preset: {preset_selection}")
        return ALL_CLAP_PRESETS[preset_selection]
    elif custom_prompts_text:
        log.info("Using prompts from custom text input.")
        prompts = [p.strip() for p in custom_prompts_text.split('\n') if p.strip()]
        return prompts
    else:
        return []

def _save_preset_action(preset_name, custom_prompts_text):
    """Handles the save preset button click."""
    if not preset_name.strip():
        return gr.update(), gr.update(value="Preset name cannot be empty.") # Update dropdown, show status
    
    prompts = [p.strip() for p in custom_prompts_text.split('\n') if p.strip()]
    if not prompts:
        return gr.update(), gr.update(value="No prompts entered to save.")

    success = preset_utils.save_clap_prompt_preset(preset_name, prompts)
    if success:
        # Reload presets and update dropdown
        global ALL_CLAP_PRESETS, PRESET_CHOICES
        ALL_CLAP_PRESETS = preset_utils.load_clap_prompt_presets()
        PRESET_CHOICES = list(ALL_CLAP_PRESETS.keys())
        # Update dropdown choices, clear preset name input, show success
        return gr.update(choices=PRESET_CHOICES), gr.update(value=f"Preset '{preset_name}' saved.")
    else:
        # Update dropdown choices (might not have changed), show error
        return gr.update(choices=PRESET_CHOICES), gr.update(value=f"Failed to save preset '{preset_name}'.")

# --- Main Analysis Function ---

def analyze_audio(audio_file_obj, 
                  separator_model_display_name, # Renamed input to reflect it's the display name
                  clap_preset_selection, 
                  clap_custom_prompts, 
                  clap_confidence_threshold, 
                  progress=gr.Progress(track_tqdm=True)):
    """The main function called by Gradio to process an uploaded audio file."""
    if audio_file_obj is None:
        return None, None, "Please upload an audio file."

    input_audio_path = Path(audio_file_obj.name)
    log.info(f"Starting analysis for: {input_audio_path}")
    
    # 1. Determine output directory
    output_dir = file_utils.generate_output_path(settings.BASE_OUTPUT_DIR, input_audio_path.name)
    temp_dir = output_dir / "temp" # Subdir for intermediate files
    file_utils.ensure_dir(temp_dir)
    
    final_json_path = output_dir / "results.json"
    status_updates = []
    
    try:
        # 2. Get Prompts
        prompts = _get_prompts_from_ui(clap_preset_selection, clap_custom_prompts)
        if not prompts:
            return None, None, "Error: No CLAP prompts provided or selected."
        status_updates.append(f"Using {len(prompts)} CLAP prompts.")

        # 3. Audio Separation
        status_updates.append(f"Separating audio using {separator_model_display_name}...")
        progress(0.1, desc="Separating audio...")

        # --- Get separator config based on display name ---
        if separator_model_display_name not in settings.AUDIO_SEPARATOR_AVAILABLE_MODELS:
            log.error(f"Selected separator model '{separator_model_display_name}' not found in configuration.")
            return None, None, f"Error: Invalid separator model selected."
        
        selected_model_config = settings.AUDIO_SEPARATOR_AVAILABLE_MODELS[separator_model_display_name]
        model_internal_name = selected_model_config['model_name']
        separator_params = selected_model_config.get('params', {}) # Get default params or empty dict
        log.info(f"Using internal separator model name: {model_internal_name} with params: {separator_params}")
        # --- End separator config ---

        # Initialize SeparatorWrapper with unpacked params
        separator = AudioSeparatorWrapper(model_name=model_internal_name,
                                        output_dir=temp_dir,
                                        **separator_params # Pass the specific params for this model
                                        )
        separated_stems = separator.separate(input_audio_path)
        if not separated_stems or "Vocals" not in separated_stems or "Instrumental" not in separated_stems:
            log.error(f"Separation failed or did not produce expected Vocal/Instrumental stems.")
            return None, None, "Error: Audio separation failed or missing expected stems."
        status_updates.append("Separation complete.")

        # 4. Resampling & Annotation
        results_data = {"original_audio": str(input_audio_path), "analysis": {}}
        annotator = CLAPAnnotatorWrapper() # Uses defaults from settings

        total_stems = len(separated_stems)
        processed_stems = 0
        for stem_name, stem_path in separated_stems.items():
            status_updates.append(f"Processing {stem_name} stem...")
            
            # 4a. Resample
            progress(0.3 + 0.6 * (processed_stems / total_stems), desc=f"Resampling {stem_name}...")
            resampled_path = temp_dir / f"{stem_path.stem}_resampled.wav"
            audio_utils.resample_audio_ffmpeg(stem_path, resampled_path, settings.CLAP_EXPECTED_SR)
            status_updates.append(f"Resampled {stem_name} to {settings.CLAP_EXPECTED_SR}Hz.")

            # 4b. Annotate
            progress(0.5 + 0.6 * (processed_stems / total_stems), desc=f"Annotating {stem_name}...")
            
            def clap_progress_callback(current_chunk, total_chunks):
                base_progress = 0.5 + 0.6 * (processed_stems / total_stems)
                chunk_progress = (current_chunk / total_chunks) * (0.6 / total_stems) # Progress within this stem's annotation phase
                progress(base_progress + chunk_progress, desc=f"Annotating {stem_name} (Chunk {current_chunk}/{total_chunks})")

            stem_results = annotator.annotate(resampled_path, 
                                              prompts, 
                                              clap_confidence_threshold,
                                              progress_callback=clap_progress_callback)
            results_data["analysis"][stem_name] = stem_results
            status_updates.append(f"Annotation complete for {stem_name}. Found {len(stem_results.get('detections',[]))} detections.")
            processed_stems += 1

        # 5. Save Results
        progress(0.95, desc="Saving results...")
        try:
            with open(final_json_path, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=4)
            log.info(f"Results saved to: {final_json_path}")
            status_updates.append("Analysis complete. Results saved.")
            return results_data, final_json_path, "\n".join(status_updates)
        except Exception as e:
            log.exception(f"Failed to save results JSON to {final_json_path}: {e}")
            status_updates.append(f"Error: Failed to save results JSON: {e}")
            return None, None, "\n".join(status_updates)

    except Exception as e:
        log.exception("An error occurred during the analysis process.")
        error_message = f"Error: {e}\n\nTraceback:\n{traceback.format_exc()}"
        status_updates.append(error_message)
        # Return None for results/file path, but include error in status
        return None, None, "\n".join(status_updates)
    
    finally:
        # 6. Cleanup Temp Files
        log.info(f"Cleaning up temporary directory: {temp_dir}")
        file_utils.cleanup_directory(temp_dir)

# --- Gradio UI Definition ---
def create_gradio_app():
    """Creates and returns the Gradio interface."""
    log.info("Creating Gradio application interface.")

    # Load initial presets
    all_available_presets = preset_utils.load_clap_prompt_presets() # Returns a Dict
    available_preset_names = list(all_available_presets.keys())
    available_separator_models = list(settings.AUDIO_SEPARATOR_AVAILABLE_MODELS.keys())

    # --- UI Components ---
    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown("# CLAP Annotator")
        status_textbox = gr.Textbox(label="Status", value=DEFAULT_STATUS, interactive=False)

        with gr.Row():
            with gr.Column(scale=1):
                # Inputs & Configuration
                audio_input = gr.Audio(label="Upload Audio File", type="filepath") # Use filepath for easier handling

                separator_model_dropdown = gr.Dropdown(
                    label="Audio Separator Model",
                    choices=available_separator_models,
                    value=settings.DEFAULT_AUDIO_SEPARATOR_MODEL, # Assuming a default is set in settings
                    info="Select the model for separating vocals/instruments."
                )

                clap_confidence_slider = gr.Slider(
                    label="CLAP Confidence Threshold",
                    minimum=0.0, maximum=1.0, step=0.01,
                    value=settings.DEFAULT_CLAP_CONFIDENCE_THRESHOLD,
                    info="Minimum confidence for a CLAP detection to be included."
                )

                clap_chunk_duration_slider = gr.Slider(
                    label="CLAP Analysis Chunk Duration (seconds)",
                    minimum=1.0, maximum=10.0, step=0.5, # Adjust range/step as needed
                    value=settings.CLAP_CHUNK_DURATION_S,
                    info="Duration of audio chunks processed by CLAP."
                )

                keep_audio_checkbox = gr.Checkbox(
                    label="Keep Separated Audio Files",
                    value=True, # Default to keeping audio
                    info="If checked, saves the separated audio stems in the output folder."
                )

                analyze_button = gr.Button("Analyze Audio", variant="primary")

            with gr.Column(scale=1):
                # Presets & Prompts
                gr.Markdown("### CLAP Prompts")
                preset_dropdown = gr.Dropdown(
                    label="Load Prompt Preset",
                    choices=available_preset_names, # Use the loaded names
                    info="Select a saved set of prompts."
                )
                prompt_textbox = gr.Textbox(
                    label="Enter Prompts (one per line)",
                    lines=8, # Increased lines for better visibility
                    info="Enter text prompts for CLAP to detect. Loaded from preset or manually entered."
                )
                with gr.Row():
                    new_preset_name_textbox = gr.Textbox(label="New Preset Name", scale=3)
                    save_preset_button = gr.Button("Save Preset", scale=1)

                # Outputs
                gr.Markdown("### Results")
                json_output = gr.JSON(label="Annotation Results")
                # Use File output for downloading the generated JSON
                download_output_file = gr.File(label="Download Results JSON")
                # Placeholder for potential audio outputs
                # audio_output_vocals = gr.Audio(label="Preserved Vocals", type="filepath", interactive=False)
                # audio_output_instrumental = gr.Audio(label="Preserved Instrumental", type="filepath", interactive=False)


        # --- UI Interactions ---

        # Function to update the prompt textbox when a preset is selected
        def update_prompts_from_preset(preset_name):
            log.debug(f"Preset selected: {preset_name}")
            if not preset_name:
                return gr.update(value="") # Clear textbox if selection is cleared
            try:
                # Use the initially loaded dictionary to get prompts for a selected name
                prompts = all_available_presets.get(preset_name, []) 
                return gr.update(value="\n".join(prompts))
            except FileNotFoundError: # Should not happen if using the preloaded dict
                log.error(f"Selected preset file not found: {preset_name}")
                return gr.update(value="") # Clear if preset file is missing
            except Exception as e:
                log.error(f"Error loading preset '{preset_name}': {e}")
                return gr.update(value="")

        # Function to save the current prompts as a new preset
        def save_preset(preset_name, prompts_text):
            preset_name = preset_name.strip()
            if not preset_name:
                log.warning("Attempted to save preset with empty name.")
                return gr.update(), gr.update(value="Preset name cannot be empty.") # Update dropdown, update status
            if not prompts_text.strip():
                log.warning(f"Attempted to save empty prompts for preset '{preset_name}'.")
                return gr.update(), gr.update(value="Cannot save empty prompts.")

            prompts_list = [p.strip() for p in prompts_text.split('\n') if p.strip()]
            if not prompts_list:
                 log.warning(f"Attempted to save preset '{preset_name}' with only whitespace prompts.")
                 return gr.update(), gr.update(value="Cannot save preset with only whitespace prompts.")

            try:
                preset_utils.save_clap_prompt_preset(preset_name, prompts_list)
                log.info(f"Preset '{preset_name}' saved successfully.")
                # Reload choices and update dropdown
                global all_available_presets, available_preset_names # Update global vars
                all_available_presets = preset_utils.load_clap_prompt_presets()
                available_preset_names = list(all_available_presets.keys())
                # Return updates for dropdown, status, and potentially clear name input
                return gr.update(choices=available_preset_names, value=preset_name), gr.update(value=f"Preset '{preset_name}' saved."), gr.update(value="") # Update dropdown, status, clear name input
            except Exception as e:
                log.error(f"Error saving preset '{preset_name}': {e}", exc_info=True)
                return gr.update(), gr.update(value=f"Error saving preset: {e}"), gr.update() # Update status, leave name input

        # Wire preset interactions
        preset_dropdown.change(
            fn=update_prompts_from_preset,
            inputs=[preset_dropdown],
            outputs=[prompt_textbox]
        )

        save_preset_button.click(
            fn=save_preset,
            inputs=[new_preset_name_textbox, prompt_textbox],
            outputs=[preset_dropdown, status_textbox, new_preset_name_textbox] # Update dropdown, status, clear name input
        )

        # TODO: Implement main analysis logic and wire the analyze_button
        def analyze_audio_wrapper(
            audio_filepath,
            separator_model_name,
            clap_prompts_text,
            clap_confidence,
            clap_chunk_duration,
            keep_separated_audio,
            progress=gr.Progress(track_tqdm=True)
        ):
            """Wrapper function to orchestrate the audio analysis pipeline."""
            if not audio_filepath:
                log.warning("No audio file provided for analysis.")
                return DEFAULT_STATUS, None, None # Status, JSON output, Download file

            status_updates = ["Starting analysis..."]
            yield "\n".join(status_updates), None, None

            try:
                input_audio_path = Path(audio_filepath)
                if not input_audio_path.exists() or not input_audio_path.is_file():
                    raise FileNotFoundError(f"Input audio file not found: {audio_filepath}")

                # --- 0. Get Prompts ---
                prompts = [p.strip() for p in clap_prompts_text.split('\n') if p.strip()]
                if not prompts:
                    status_updates.append("Error: No CLAP prompts provided. Please enter prompts or select a preset.")
                    log.error("No CLAP prompts provided for analysis.")
                    yield "\n".join(status_updates), None, None
                    return
                status_updates.append(f"Using CLAP prompts: {prompts}")
                yield "\n".join(status_updates), None, None

                # --- 1. Setup Output Directory ---
                progress(0.05, desc="Setting up output directory...")
                output_dir = file_utils.generate_output_path(settings.BASE_OUTPUT_DIR, input_audio_path.name)
                temp_dir = output_dir / "temp_processing"
                file_utils.ensure_dir(temp_dir)
                status_updates.append(f"Output will be saved to: {output_dir}")
                log.info(f"Output directory: {output_dir}, Temp directory: {temp_dir}")
                yield "\n".join(status_updates), None, None

                # --- 2. Initialize Wrappers ---
                progress(0.1, desc="Initializing models...")
                status_updates.append("Initializing audio separator...")
                yield "\n".join(status_updates), None, None
                separator_model_config = settings.AUDIO_SEPARATOR_AVAILABLE_MODELS.get(separator_model_name)
                if not separator_model_config:
                    raise ValueError(f"Separator model '{separator_model_name}' not found in settings.")
                
                separator = AudioSeparatorWrapper(
                    model_name=separator_model_config["model_name"],
                    model_file_dir=settings.AUDIO_SEPARATOR_MODEL_DIR,
                    output_dir=temp_dir # Separate into temp_dir
                )
                status_updates.append("Audio separator initialized.")

                status_updates.append("Initializing CLAP annotator...")
                yield "\n".join(status_updates), None, None
                annotator = CLAPAnnotatorWrapper(
                    model_name=settings.CLAP_MODEL_NAME,
                    expected_sr=settings.CLAP_EXPECTED_SR,
                    chunk_duration_s=clap_chunk_duration
                    # confidence_threshold is passed to process_audio, not init
                )
                status_updates.append("CLAP annotator initialized.")
                yield "\n".join(status_updates), None, None

                # --- 3. Audio Separation ---
                progress(0.2, desc="Separating audio...")
                status_updates.append(f"Separating audio using {separator_model_name}...")
                yield "\n".join(status_updates), None, None
                
                # Determine expected stems from model config
                expected_stems_from_config = separator_model_config.get("expected_stems", ["vocals", "instrumental"])
                log.info(f"Expected stems for model '{separator_model_name}': {expected_stems_from_config}")

                separated_stem_paths = separator.separate(input_audio_path)

                if not separated_stem_paths:
                    status_updates.append("Error: Audio separation failed to produce any output files.")
                    log.error("Audio separation failed.")
                    yield "\n".join(status_updates), None, None
                    return
                
                status_updates.append(f"Audio separated. Found stems: {list(separated_stem_paths.keys())}") # Use .keys() if it's a dict
                log.info(f"Separated stems: {separated_stem_paths}")
                yield "\n".join(status_updates), None, None

                # --- 4. Resampling and Annotation per Stem ---
                all_results = {
                    "input_file": str(input_audio_path.name),
                    "output_directory": str(output_dir.relative_to(Path.cwd()) if output_dir.is_relative_to(Path.cwd()) else output_dir),
                    "separator_model": separator_model_name,
                    "clap_model": settings.CLAP_MODEL_NAME,
                    "clap_prompts": prompts,
                    "clap_confidence_threshold": clap_confidence,
                    "clap_chunk_duration_s": clap_chunk_duration,
                    "analysis_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "annotations": {},
                    "preserved_audio_files": {}
                }

                num_stems_to_process = len(separated_stem_paths)
                progress_per_stem_annotation = 0.6 / num_stems_to_process if num_stems_to_process > 0 else 0
                current_progress = 0.3 # After separation

                for i, (stem_name, stem_path_str) in enumerate(separated_stem_paths.items()):
                    stem_path = Path(stem_path_str)
                    if not stem_path.exists():
                        status_updates.append(f"Warning: Separated stem file not found: {stem_path}. Skipping.")
                        log.warning(f"Separated stem file not found: {stem_path}. Skipping.")
                        yield "\n".join(status_updates), None, None
                        continue

                    status_updates.append(f"Processing stem: {stem_name}...")
                    yield "\n".join(status_updates), None, None

                    # Resample
                    progress(current_progress, desc=f"Resampling {stem_name}...")
                    status_updates.append(f"Resampling {stem_name} to {settings.CLAP_EXPECTED_SR}Hz...")
                    yield "\n".join(status_updates), None, None
                    resampled_path = temp_dir / f"{stem_path.stem}_resampled_for_clap.wav"
                    audio_utils.resample_audio_ffmpeg(stem_path, resampled_path, settings.CLAP_EXPECTED_SR)
                    status_updates.append(f"Resampled {stem_name} to {resampled_path}")
                    log.info(f"Resampled {stem_name} from {stem_path} to {resampled_path}")
                    yield "\n".join(status_updates), None, None

                    # Annotate
                    status_updates.append(f"Annotating {stem_name} with CLAP...")
                    yield "\n".join(status_updates), None, None
                    
                    # Define a callback for CLAP progress
                    def clap_progress_callback(chunk_idx, total_chunks):
                        chunk_progress = (chunk_idx + 1) / total_chunks
                        stem_specific_progress = current_progress + (progress_per_stem_annotation * chunk_progress)
                        progress(stem_specific_progress, desc=f"CLAP: {stem_name} ({chunk_idx+1}/{total_chunks})")

                    stem_annotations = annotator.annotate( # Corrected method name
                        audio_path=resampled_path,
                        text_prompts=prompts, # Corrected argument name
                        confidence_threshold=clap_confidence,
                        progress_callback=clap_progress_callback
                    )
                    all_results["annotations"][stem_name] = stem_annotations
                    status_updates.append(f"Annotation complete for {stem_name}. Detections: {len(stem_annotations)}")
                    log.info(f"Annotation for {stem_name} yielded {len(stem_annotations)} detections.")
                    yield "\n".join(status_updates), all_results, None # Show intermediate JSON results
                    
                    current_progress += progress_per_stem_annotation
                
                progress(0.9, desc="Finalizing results...")

                # --- 5. Save Results & Preserve Audio ---
                final_json_path = output_dir / "annotations.json"
                with open(final_json_path, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2)
                status_updates.append(f"Results saved to: {final_json_path}")
                log.info(f"Final results saved to {final_json_path}")

                if keep_separated_audio:
                    status_updates.append("Preserving separated audio files...")
                    yield "\n".join(status_updates), all_results, None
                    for stem_name, original_stem_path_str in separated_stem_paths.items():
                        original_stem_path = Path(original_stem_path_str)
                        if original_stem_path.exists():
                            preserved_filename = f"{input_audio_path.stem}_{stem_name}{original_stem_path.suffix}"
                            preserved_path = output_dir / preserved_filename
                            shutil.copy2(original_stem_path, preserved_path)
                            all_results["preserved_audio_files"][stem_name] = str(preserved_path.relative_to(Path.cwd()) if preserved_path.is_relative_to(Path.cwd()) else preserved_path)
                            status_updates.append(f"Preserved {stem_name} to {preserved_path}")
                            log.info(f"Preserved {stem_name} from {original_stem_path} to {preserved_path}")
                        else:
                             status_updates.append(f"Warning: Original stem file {original_stem_path} not found for preservation.")
                             log.warning(f"Original stem file {original_stem_path} not found for preservation.")
                    # Re-save JSON with preserved audio paths if any were added
                    with open(final_json_path, 'w', encoding='utf-8') as f:
                        json.dump(all_results, f, indent=2)
                yield "\n".join(status_updates), all_results, str(final_json_path)

                status_updates.append("Analysis complete!")
                progress(1.0, desc="Analysis complete!")
                log.info("Audio analysis pipeline completed successfully.")
                return "\n".join(status_updates), all_results, str(final_json_path)

            except Exception as e:
                log.error(f"Error during audio analysis: {e}", exc_info=True)
                error_message = f"An error occurred: {type(e).__name__}: {e}\n{traceback.format_exc()}"
                status_updates.append(f"ERROR: {error_message}")
                # Ensure some output for JSON if an error occurs, perhaps with error info
                error_json = {"error": error_message, "status_log": "\n".join(status_updates)}
                yield "\n".join(status_updates), error_json, None 
                return "\n".join(status_updates), error_json, None
            finally:
                # --- 6. Cleanup (if temp_dir was created) ---
                if 'temp_dir' in locals() and temp_dir.exists():
                    try:
                        file_utils.cleanup_directory(temp_dir) # Corrected function name
                        status_updates.append(f"Temporary files cleaned up from {temp_dir}.")
                        log.info(f"Cleaned up temporary directory: {temp_dir}")
                    except Exception as e:
                        log.error(f"Error during temporary file cleanup: {e}", exc_info=True)
                        status_updates.append(f"Error cleaning up temporary files: {e}")
                # Yield final status if cleanup happened after main return
                # This part might not be visible if a return already happened in try/except
                # For robustness, the main return points should handle final status.

        # Wire the analyze button
        analyze_button.click(
            fn=analyze_audio_wrapper,
            inputs=[
                audio_input,
                separator_model_dropdown,
                prompt_textbox, # Changed from preset_dropdown to use the actual text
                clap_confidence_slider,
                clap_chunk_duration_slider,
                keep_audio_checkbox
            ],
            outputs=[
                status_textbox,
                json_output,
                download_output_file
            ]
        )

    return app

# --- Main Execution ---
if __name__ == "__main__":
    log.info("Starting Gradio App...")
    # Check for ffmpeg? (Requires subprocess)
    # try:
    #     subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    #     log.info("ffmpeg found.")
    # except (FileNotFoundError, subprocess.CalledProcessError) as e:
    #     log.warning("ffmpeg command not found or failed. Resampling will likely fail.")
    #     # Optionally raise error or show warning in UI later
        
    app_ui = create_gradio_app()
    app_ui.launch() # Add share=True for public link if needed 