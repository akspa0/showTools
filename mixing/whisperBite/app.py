import gradio as gr
import os
from whisperBite import process_audio
from utils import download_audio

def run_pipeline(input_file, input_folder, url, output_folder, model, num_speakers, enable_vocal_separation):
    """Run the audio processing pipeline based on user inputs."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # Determine input source
    if url:
        input_path = download_audio(url, output_folder)
    elif input_file:
        input_path = input_file
    elif input_folder:
        input_path = input_folder
    else:
        return "Please provide an input file, folder, or URL."

    # Run the processing pipeline
    try:
        process_audio(
            input_path=input_path,
            output_dir=output_folder,
            model_name=model,
            enable_vocal_separation=enable_vocal_separation,
            num_speakers=num_speakers
        )
        return f"Processing complete! Results saved to {output_folder}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Gradio interface
def build_interface():
    def set_hf_token(hf_token):
        os.environ['HF_TOKEN'] = hf_token
        return "Hugging Face token set successfully."

    with gr.Blocks() as demo:
        gr.Markdown("# whisperBite")
        gr.Markdown("This tool processes audio files into transcribed soundbites using Whisper and pyannote.audio.")

        with gr.Row():
            input_file = gr.File(label="Input File", file_types=[".wav", ".mp3", ".m4a"])
            input_folder = gr.Textbox(label="Input Folder", placeholder="Path to folder containing audio files")
            url = gr.Textbox(label="URL", placeholder="Provide a URL to download audio")

        output_folder = gr.Textbox(label="Output Folder", placeholder="Path to save output files")

        with gr.Row():
            model = gr.Dropdown(label="Whisper Model", choices=["base", "small", "medium", "large", "turbo"], value="turbo")
            num_speakers = gr.Slider(label="Number of Speakers", minimum=1, maximum=10, step=1, value=2)

        enable_vocal_separation = gr.Checkbox(label="Enable Vocal Separation", value=False)

        hf_token = gr.Textbox(label="Hugging Face Token", placeholder="Set your Hugging Face token for pyannote.audio")
        set_token_button = gr.Button("Set Token")
        submit_button = gr.Button("Run Pipeline")

        output_message = gr.Textbox(label="Output Message", interactive=False)

        set_token_button.click(
            fn=set_hf_token,
            inputs=[hf_token],
            outputs=[output_message]
        )

        submit_button.click(
            fn=run_pipeline,
            inputs=[input_file, input_folder, url, output_folder, model, num_speakers, enable_vocal_separation],
            outputs=output_message
        )

    return demo

if __name__ == "__main__":
    interface = build_interface()
    interface.launch()
