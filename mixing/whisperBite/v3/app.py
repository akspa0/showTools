import gradio as gr
import os
import json
from whisperBite import process_audio
from utils import download_audio

def run_pipeline(input_file, input_folder, url, output_folder, model, num_speakers, 
                auto_speakers, enable_vocal_separation, hf_token):
    """Run the audio processing pipeline based on user inputs."""
    # Set Hugging Face token if provided
    if hf_token:
        os.environ['HF_TOKEN'] = hf_token

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
            num_speakers=num_speakers,
            auto_speakers=auto_speakers
        )
        
        # Find the results zip file
        result_files = []
        for root, _, files in os.walk(output_folder):
            for file in files:
                if file.endswith("_results.zip"):
                    result_files.append(os.path.join(root, file))
                elif file == "master_transcript.txt":
                    with open(os.path.join(root, file), 'r') as f:
                        transcript = f.read()
        
        results_message = f"Processing complete! Results saved to {output_folder}\n"
        if result_files:
            results_message += f"\nZip files created:\n" + "\n".join(result_files)
        
        return results_message, result_files[0] if result_files else None, transcript if 'transcript' in locals() else ""
    except Exception as e:
        import traceback
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
                        input_file = gr.File(label="Input Audio File", file_types=[".wav", ".mp3", ".m4a"])
                    with gr.Column():
                        input_folder = gr.Textbox(label="Input Folder Path", placeholder="Path to folder containing audio files")
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
                        value="base",
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
                
                hf_token = gr.Textbox(
                    label="Hugging Face Token (Required)", 
                    placeholder="Set your Hugging Face token for pyannote.audio",
                    type="password",
                    info="Get token at huggingface.co/settings/tokens"
                )
        
        submit_button = gr.Button("Process Audio", variant="primary")
        
        with gr.Row():
            output_message = gr.Textbox(label="Status", interactive=False)
            result_file = gr.File(label="Download Results")
            transcript_preview = gr.TextArea(label="Transcript Preview", interactive=False, lines=10)

        submit_button.click(
            fn=run_pipeline,
            inputs=[
                input_file, input_folder, url, output_folder, model, 
                num_speakers, auto_speakers, enable_vocal_separation, hf_token
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
    interface.launch(share=args.public, server_port=args.port)
