import os
os.environ["GRADIO_SERVER_MAX_REQUEST_SIZE"] = "1024"  # 1GB
import gradio as gr
import os as _os
from core import mhrp_pipeline

# Default options
DEFAULT_CLAP_PROMPTS = "telephone noises, dial tone, ring tone, telephone interference"
DEFAULT_SEPARATOR_MODEL = "Mel Band RoFormer Vocals"
DEFAULT_CONFIDENCE = 0.6
DEFAULT_WHISPER_MODEL = "large-v3"
DEFAULT_NUM_SPEAKERS = 2
DEFAULT_TARGET_LUFS = -14.0
DEFAULT_NUM_WORKERS = 2

# Add workflow_preset to the UI
WORKFLOW_PRESETS = [
    ("show-edit", "Show-Edit (compile all calls into a single show, insert tones.wav only between calls, not at end of calls)")
    # Future: add more presets here
]

# Process function for Gradio

def process(files, folder_path, mode, workflow_preset, prompts, separator_model, confidence, add_tones, normalize, target_lufs, num_workers, whisper_model, num_speakers, process_left_channel, generate_image_prompts):
    output_dir = "mhrpTools_Output"
    _os.makedirs(output_dir, exist_ok=True)
    results = []
    skipped_files = []
    # If show-edit mode, always enable image prompt generation
    if workflow_preset == 'show-edit':
        generate_image_prompts = True
    # Build options dict to pass to pipeline
    options = dict(
        prompts=prompts,
        separator_model=separator_model,
        confidence=confidence,
        add_tones=add_tones,
        normalize=normalize,
        target_lufs=target_lufs,
        num_workers=num_workers,
        whisper_model=whisper_model,
        num_speakers=num_speakers,
        process_left_channel=process_left_channel,
        workflow_preset=workflow_preset,
        generate_image_prompts=generate_image_prompts
    )
    # If folder path is provided, process as a batch job
    if folder_path and _os.path.isdir(folder_path):
        skipped = mhrp_pipeline(folder_path, output_dir, mode=mode, **options)
        results.append(f"Processed folder: {folder_path}")
        if skipped:
            skipped_files.extend(skipped)
    # If files are uploaded, process each one
    elif files:
        for file in files:
            skipped = mhrp_pipeline(file.name, output_dir, mode=mode, **options)
            results.append(f"Processed file: {os.path.basename(file.name)}")
            if skipped:
                skipped_files.extend(skipped)
    else:
        return "Please upload files or specify a folder path."
    out = "\n".join(results) + f"\nResults in {output_dir}"
    if skipped_files:
        out += f"\nSkipped files (invalid/empty):\n" + "\n".join(skipped_files)
    return out

def main():
    with gr.Blocks() as demo:
        gr.Markdown("# mhrpTools: Unified Audio Processing")
        with gr.Row():
            input_files = gr.Files(label="Input Audio Files (multiple)")
            folder_path = gr.Textbox(label="Input Folder Path (for batch jobs)", placeholder="/path/to/folder")
        workflow_preset = gr.Dropdown([p[0] for p in WORKFLOW_PRESETS], value="show-edit", label="Workflow Preset", info="Select the overall workflow. 'Show-Edit' compiles all calls into a single show, inserting tones.wav only between calls (not at the end of each call). This is ideal for show compilation and future splitting.")
        mode = gr.Radio(["auto", "mixing", "soundbites", "both", "arbitrary-audio"], value="auto", label="Processing Mode", info="Choose the main workflow mode.")
        with gr.Tab("ClapAnnotator Options"):
            prompts = gr.Textbox(label="CLAP Prompts", value=DEFAULT_CLAP_PROMPTS, info="Comma-separated prompts for CLAP annotation.")
            separator_model = gr.Dropdown(["Mel Band RoFormer Vocals", "UVR-MDX-NET Main", "Demucs"], value=DEFAULT_SEPARATOR_MODEL, label="Separator Model", info="Audio separation model.")
            confidence = gr.Slider(0.0, 1.0, value=DEFAULT_CONFIDENCE, step=0.01, label="CLAP Confidence Threshold", info="Minimum confidence for CLAP detections.")
        with gr.Tab("preFapMix Options"):
            add_tones = gr.Checkbox(label="Append Tones to Stereo Outputs", value=False, info="Add tones to the end of each stereo file after mixing.")
            normalize = gr.Checkbox(label="Enable Loudness Normalization", value=False, info="Normalize audio loudness before mixing.")
            target_lufs = gr.Slider(-30, -10, value=DEFAULT_TARGET_LUFS, step=0.5, label="Target LUFS", info="Target loudness for normalization.")
            num_workers = gr.Slider(1, 8, value=DEFAULT_NUM_WORKERS, step=1, label="Number of Workers", info="Number of workers for preFapMix transcription.")
        with gr.Tab("WhisperBite Options"):
            whisper_model = gr.Dropdown(["large-v3", "medium", "small", "base"], value=DEFAULT_WHISPER_MODEL, label="Whisper Model", info="Model for transcription.")
            num_speakers = gr.Slider(1, 10, value=DEFAULT_NUM_SPEAKERS, step=1, label="Number of Speakers", info="Number of speakers for diarization.")
            process_left_channel = gr.Checkbox(label="Process Left Channel (recv_out) for Soundbiting", value=False, info="Also process recv_out files for soundbiting.")
        with gr.Tab("AI Image Prompts"):
            generate_image_prompts = gr.Checkbox(label="Generate AI Image Prompt for Each Audio", value=True, info="If enabled, generates a detailed image prompt for each audio transcript using the LLM. Always enabled for show-edit mode.")
        run_btn = gr.Button("Run Pipeline")
        output = gr.Textbox(label="Status/Log", lines=12)

        def on_run(files, folder_path, mode, workflow_preset, prompts, separator_model, confidence, add_tones, normalize, target_lufs, num_workers, whisper_model, num_speakers, process_left_channel, generate_image_prompts):
            return process(files, folder_path, mode, workflow_preset, prompts, separator_model, confidence, add_tones, normalize, target_lufs, num_workers, whisper_model, num_speakers, process_left_channel, generate_image_prompts)

        run_btn.click(
            on_run,
            inputs=[input_files, folder_path, mode, workflow_preset, prompts, separator_model, confidence, add_tones, normalize, target_lufs, num_workers, whisper_model, num_speakers, process_left_channel, generate_image_prompts],
            outputs=output
        )
    demo.launch()

if __name__ == "__main__":
    main() 