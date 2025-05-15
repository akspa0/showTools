import gradio as gr
from pathlib import Path
from .pipeline import UnifiedPipeline
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / 'ClapAnnotator'))
from ClapAnnotator.config.settings import AUDIO_SEPARATOR_AVAILABLE_MODELS, DEFAULT_AUDIO_SEPARATOR_MODEL
import os
import yaml
from . import list_pipeline_presets, load_pipeline_preset, save_pipeline_preset

# --- Preset Management ---
PRESET_DIR = Path(__file__).resolve().parent.parent / 'ClapAnnotator' / '_presets' / 'clap_prompts'
def list_presets():
    if not PRESET_DIR.exists():
        return []
    return [f.stem for f in PRESET_DIR.glob('*.txt')]
def load_preset(name):
    path = PRESET_DIR / f"{name}.txt"
    if path.exists():
        return path.read_text(encoding='utf-8')
    return ''
def save_preset(name, prompts):
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    (PRESET_DIR / f"{name}.txt").write_text(prompts, encoding='utf-8')

def run_pipeline(input_type, input_path, input_url, show_name, output_dir,
                 separator_model,
                 clap_model, clap_chunk_duration, clap_confidence, clap_prompts,
                 mix_normalize, mix_tones, mix_lufs,
                 wb_model, wb_num_speakers, wb_enable_word_extraction, wb_enable_second_pass, wb_auto_speakers, wb_attempt_sound_detection):
    try:
        pipeline = UnifiedPipeline(
            show_name=show_name,
            output_dir=Path(output_dir),
            separator_model=separator_model,
            clap_model=clap_model or None,
            clap_chunk_duration=int(clap_chunk_duration) if clap_chunk_duration else None,
            clap_confidence=float(clap_confidence) if clap_confidence else None,
            clap_prompts=[p.strip() for p in clap_prompts.split(',')] if clap_prompts else None,
            mix_normalize=mix_normalize,
            mix_tones=mix_tones,
            mix_lufs=float(mix_lufs) if mix_lufs else None,
            wb_model=wb_model or None,
            wb_num_speakers=int(wb_num_speakers) if wb_num_speakers else None,
            wb_enable_word_extraction=wb_enable_word_extraction,
            wb_enable_second_pass=wb_enable_second_pass,
            wb_auto_speakers=wb_auto_speakers,
            wb_attempt_sound_detection=wb_attempt_sound_detection
        )
        if input_type == "URL":
            input_value = input_url
        else:
            input_value = input_path
        if not input_value or not show_name or not output_dir:
            return "Error: Input, show name, and output directory are required."
        pipeline.process_audio_file(input_value)
        pipeline.finalize_show()
        return f"Show '{show_name}' processed. Output in {output_dir}."
    except Exception as e:
        return f"[Pipeline Error] {e}"

def launch_ui():
    separator_model_choices = list(AUDIO_SEPARATOR_AVAILABLE_MODELS.keys())
    with gr.Blocks() as demo:
        gr.Markdown("# Unified mhrpTools Pipeline")
        # --- Pipeline Preset UI ---
        pipeline_preset_choices = list_pipeline_presets()
        pipeline_preset_dropdown = gr.Dropdown(pipeline_preset_choices, label="Pipeline Preset", info="Select a pipeline preset to load all config fields", interactive=True)
        load_pipeline_preset_btn = gr.Button("Load Pipeline Preset")
        save_pipeline_preset_name = gr.Textbox(label="Save Current Config as Pipeline Preset Name")
        save_pipeline_preset_btn = gr.Button("Save Pipeline Preset")

        with gr.Tab("Core"):
            input_type = gr.Radio(["File/Folder", "URL"], value="File/Folder", label="Input Type")
            input_path = gr.Textbox(label="Input File/Folder Path", info="Path to audio/video file or directory")
            input_url = gr.Textbox(label="Input URL", info="YouTube or direct link", visible=False)
            def toggle_input_type(choice):
                return gr.update(visible=choice=="File/Folder"), gr.update(visible=choice=="URL")
            input_type.change(toggle_input_type, inputs=input_type, outputs=[input_path, input_url])
            show_name = gr.Textbox(label="Show Name", info="Name for the show")
            output_dir = gr.Textbox(label="Output Directory", info="Where to save outputs")
        with gr.Tab("Separation"):
            separator_model = gr.Dropdown(separator_model_choices, value=DEFAULT_AUDIO_SEPARATOR_MODEL, label="Vocal Separator Model", info="Choose the model for vocal separation")
        with gr.Tab("CLAP Annotation"):
            clap_model = gr.Textbox(label="CLAP Model Name", info="HuggingFace model name (default: laion/clap-htsat-fused)")
            clap_chunk_duration = gr.Number(label="CLAP Chunk Duration (s)", info="Duration of audio chunks for CLAP", precision=0)
            clap_confidence = gr.Number(label="CLAP Confidence Threshold", info="Minimum probability for detection", precision=2)
            # --- Preset UI ---
            preset_choices = list_presets()
            preset_dropdown = gr.Dropdown(preset_choices, label="CLAP Prompt Preset", info="Select a preset or enter prompts below", interactive=True)
            load_preset_btn = gr.Button("Load Preset")
            save_preset_name = gr.Textbox(label="Save Current Prompts as Preset Name")
            save_preset_btn = gr.Button("Save Preset")
            clap_prompts = gr.Textbox(label="CLAP Prompts (comma-separated)", info="e.g. telephone noises, dial tone, ring tone")
            def do_load_preset(name):
                return load_preset(name)
            def do_save_preset(name, prompts):
                save_preset(name, prompts)
                return gr.update(choices=list_presets())
            load_preset_btn.click(do_load_preset, inputs=preset_dropdown, outputs=clap_prompts)
            save_preset_btn.click(do_save_preset, inputs=[save_preset_name, clap_prompts], outputs=preset_dropdown)
        with gr.Tab("Mixing"):
            mix_normalize = gr.Checkbox(label="Enable Loudness Normalization", value=True)
            mix_tones = gr.Checkbox(label="Insert tones.wav between calls", value=False)
            mix_lufs = gr.Number(label="Target LUFS", value=-14.0, precision=2)
        with gr.Tab("Transcription"):
            wb_model = gr.Textbox(label="WhisperBite Model Name", info="e.g. large-v3")
            wb_num_speakers = gr.Number(label="Number of Speakers", value=2, precision=0)
            wb_enable_word_extraction = gr.Checkbox(label="Enable Word Extraction", value=False)
            wb_enable_second_pass = gr.Checkbox(label="Enable Second Pass Diarization", value=False)
            wb_auto_speakers = gr.Checkbox(label="Auto Speaker Count Detection", value=False)
            wb_attempt_sound_detection = gr.Checkbox(label="Enable Sound Event Detection", value=False)
        run_btn = gr.Button("Run Pipeline")
        output = gr.Textbox(label="Status/Output")

        # --- Pipeline Preset Logic ---
        def do_load_pipeline_preset(name):
            config = load_pipeline_preset(name)
            # Return updates for all fields in order
            return (
                config.get('input_type', 'File/Folder'),
                config.get('input_path', ''),
                config.get('input_url', ''),
                config.get('show_name', ''),
                config.get('output_dir', ''),
                config.get('separator_model', DEFAULT_AUDIO_SEPARATOR_MODEL),
                config.get('clap_model', ''),
                config.get('clap_chunk_duration', 10),
                config.get('clap_confidence', 0.5),
                config.get('clap_prompts', ''),
                config.get('mix_normalize', True),
                config.get('mix_tones', False),
                config.get('mix_lufs', -14.0),
                config.get('wb_model', ''),
                config.get('wb_num_speakers', 2),
                config.get('wb_enable_word_extraction', False),
                config.get('wb_enable_second_pass', False),
                config.get('wb_auto_speakers', False),
                config.get('wb_attempt_sound_detection', False)
            )
        def do_save_pipeline_preset(name, *fields):
            config = {
                'input_type': fields[0],
                'input_path': fields[1],
                'input_url': fields[2],
                'show_name': fields[3],
                'output_dir': fields[4],
                'separator_model': fields[5],
                'clap_model': fields[6],
                'clap_chunk_duration': fields[7],
                'clap_confidence': fields[8],
                'clap_prompts': fields[9],
                'mix_normalize': fields[10],
                'mix_tones': fields[11],
                'mix_lufs': fields[12],
                'wb_model': fields[13],
                'wb_num_speakers': fields[14],
                'wb_enable_word_extraction': fields[15],
                'wb_enable_second_pass': fields[16],
                'wb_auto_speakers': fields[17],
                'wb_attempt_sound_detection': fields[18]
            }
            save_pipeline_preset(name, config)
            return gr.update(choices=list_pipeline_presets())
        load_pipeline_preset_btn.click(
            do_load_pipeline_preset,
            inputs=pipeline_preset_dropdown,
            outputs=[input_type, input_path, input_url, show_name, output_dir, separator_model, clap_model, clap_chunk_duration, clap_confidence, clap_prompts, mix_normalize, mix_tones, mix_lufs, wb_model, wb_num_speakers, wb_enable_word_extraction, wb_enable_second_pass, wb_auto_speakers, wb_attempt_sound_detection]
        )
        save_pipeline_preset_btn.click(
            do_save_pipeline_preset,
            inputs=[save_pipeline_preset_name, input_type, input_path, input_url, show_name, output_dir, separator_model, clap_model, clap_chunk_duration, clap_confidence, clap_prompts, mix_normalize, mix_tones, mix_lufs, wb_model, wb_num_speakers, wb_enable_word_extraction, wb_enable_second_pass, wb_auto_speakers, wb_attempt_sound_detection],
            outputs=pipeline_preset_dropdown
        )

        run_btn.click(
            run_pipeline,
            inputs=[input_type, input_path, input_url, show_name, output_dir,
                    separator_model,
                    clap_model, clap_chunk_duration, clap_confidence, clap_prompts,
                    mix_normalize, mix_tones, mix_lufs,
                    wb_model, wb_num_speakers, wb_enable_word_extraction, wb_enable_second_pass, wb_auto_speakers, wb_attempt_sound_detection],
            outputs=output)
    demo.launch()

if __name__ == "__main__":
    launch_ui() 