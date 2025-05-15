import gradio as gr
import os
from datetime import datetime
import logging
from preFapMix import process_audio_files

logging.basicConfig(level=logging.INFO)

def process_audio(input_dir, output_dir, transcribe, transcribe_left, transcribe_right, append_tones, normalize, target_lufs, num_workers):
    try:
        transcribe_left_flag = transcribe or transcribe_left
        transcribe_right_flag = transcribe or transcribe_right

        # Call the processing function from the script
        process_audio_files(
            input_dir,
            output_dir,
            transcribe_left=transcribe_left_flag,
            transcribe_right=transcribe_right_flag,
            append_tones=append_tones,
            normalize_audio=normalize,
            target_lufs=target_lufs,
            num_workers=num_workers
        )

        return f"Processing completed! Outputs are saved in: {output_dir}"
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        return f"Error: {e}"

def create_output_dir(base_dir="output"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"run_{timestamp}")

# Define the Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Audio Processing App")
    gr.Markdown("Upload or specify an input directory, set options, and process your audio files interactively.")

    input_dir = gr.Textbox(label="Input Directory", placeholder="Path to input audio files", lines=1)
    output_dir = gr.Textbox(label="Output Directory", value=create_output_dir(), placeholder="Path to save processed files", lines=1)
    transcribe = gr.Checkbox(label="Enable Transcription for Both Channels", value=False)
    transcribe_left = gr.Checkbox(label="Transcribe Left Channel Only", value=False)
    transcribe_right = gr.Checkbox(label="Transcribe Right Channel Only", value=False)
    append_tones = gr.Checkbox(label="Append Tones to Stereo Outputs", value=False)
    normalize = gr.Checkbox(label="Enable Loudness Normalization", value=False)
    target_lufs = gr.Slider(label="Target LUFS", minimum=-30, maximum=-10, step=0.5, value=-14.0)
    num_workers = gr.Slider(label="Number of Workers for Transcription", minimum=1, maximum=8, step=1, value=2)

    process_button = gr.Button("Start Processing")
    result = gr.Textbox(label="Result", interactive=False, lines=4)

    process_button.click(
        process_audio,
        inputs=[
            input_dir, output_dir, transcribe, transcribe_left, transcribe_right, append_tones, normalize, target_lufs, num_workers
        ],
        outputs=result
    )

app.launch()
