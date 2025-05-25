import os
import json
import string
from pathlib import Path
from extension_base import ExtensionBase
from llm_utils import run_llm_task
import argparse
from pydub import AudioSegment

"""
Usage:
    python character_persona_builder.py <output_root> [--llm-config <config_path>]

For each call_id in <output_root>/speakers/, this script merges all utterances from all speakers in each channel (left-vocals, right-vocals, or conversation),
builds a single transcript per channel, and generates a Character.AI persona in Markdown format using the embedded guidelines as the system prompt and a concise user prompt referencing the LennyBot style (but not including its content).
Outputs to <output_root>/characters/<call_title or call_id>/<channel>/.
"""

# --- EMBEDDED SYSTEM PROMPT (Character.AI guidelines) ---
CHARACTER_AI_GUIDELINES = r'''
# Character.AI Creation Guidelines

[...full content of character_ai_guidelines.md goes here...]
'''

LENNYBOT_STYLE_NOTE = (
    "in the style and structure of the LennyBot example (do not copy its content, but use its format and level of detail)"
)

def find_channels(call_folder):
    channels = []
    for d in call_folder.iterdir():
        if d.is_dir() and d.name in ["left-vocals", "right-vocals", "conversation"]:
            channels.append(d.name)
    # Prefer left/right if present
    if "left-vocals" in channels and "right-vocals" in channels:
        return ["left-vocals", "right-vocals"]
    elif "conversation" in channels:
        return ["conversation"]
    return []

def collect_utterances_merged(call_folder, channels):
    utterances = {ch: [] for ch in channels}
    for channel in channels:
        channel_dir = call_folder / channel
        if not channel_dir.exists():
            continue
        for speaker_folder in channel_dir.iterdir():
            if not speaker_folder.is_dir() or not speaker_folder.name.startswith('S'):
                continue
            speaker_id = speaker_folder.name
            for txt_file in speaker_folder.glob('*.txt'):
                base = txt_file.stem
                wav_file = txt_file.with_suffix('.wav')
                json_file = txt_file.with_suffix('.json')
                timestamp = None
                if json_file.exists():
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                        timestamp = meta.get('start', None)
                    except Exception:
                        pass
                if timestamp is None:
                    parts = base.split('-')
                    if len(parts) > 1 and parts[1].isdigit():
                        timestamp = float(parts[1])
                utter = {
                    'channel': channel,
                    'speaker_id': speaker_id,
                    'txt_file': txt_file,
                    'wav_file': wav_file if wav_file.exists() else None,
                    'timestamp': timestamp,
                    'text': txt_file.read_text(encoding='utf-8').strip()
                }
                utterances[channel].append(utter)
    for ch in utterances:
        utterances[ch].sort(key=lambda u: (u['timestamp'] if u['timestamp'] is not None else 0))
    return utterances

def create_audio_clips(utterances, out_dir, char_label):
    wavs = [u['wav_file'] for u in utterances if u['wav_file'] is not None]
    if not wavs:
        return []
    combined = AudioSegment.empty()
    for wav in wavs:
        try:
            seg = AudioSegment.from_wav(wav)
            combined += seg
        except Exception:
            continue
    durations = [15, 30, 60]
    outputs = []
    for sec in durations:
        if len(combined) == 0:
            break
        clip = combined[:sec*1000]
        out_mp3 = out_dir / f"{char_label}_{sec}s.mp3"
        clip.export(out_mp3, format="mp3")
        outputs.append(out_mp3)
    return outputs

def sanitize_title(title):
    title = title.translate(str.maketrans('', '', string.punctuation))
    title = '_'.join(title.strip().split())
    return title[:48] or None

class CharacterPersonaBuilder(ExtensionBase):
    def __init__(self, output_root, llm_config_path=None):
        super().__init__(output_root)
        self.llm_config_path = llm_config_path or 'workflows/llm_tasks.json'
        self.llm_config = self._load_llm_config()

    def _load_llm_config(self):
        config_path = Path(self.llm_config_path)
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            if 'lm_studio_base_url' in config:
                return config
            elif 'llm_tasks' in config and 'llm_config' in config:
                return config['llm_config']
        return {
            'lm_studio_base_url': 'http://localhost:1234/v1',
            'lm_studio_api_key': 'lm-studio',
            'lm_studio_model_identifier': 'llama-3.1-8b-supernova-etherealhermes',
            'lm_studio_temperature': 0.5,
            'lm_studio_max_tokens': 2048
        }

    def get_call_title(self, call_id):
        llm_dir = self.output_root / 'llm' / call_id
        title_file = llm_dir / 'call_title.txt'
        if title_file.exists():
            title = title_file.read_text(encoding='utf-8').strip()
            sanitized = sanitize_title(title)
            if sanitized:
                return sanitized
        return call_id

    def run(self):
        speakers_root = self.output_root / 'speakers'
        characters_root = self.output_root / 'characters'
        characters_root.mkdir(exist_ok=True)
        if not speakers_root.exists():
            self.log("No speakers directory found.")
            return
        callid_to_title = {}
        for call_folder in speakers_root.iterdir():
            if not call_folder.is_dir():
                continue
            call_id = call_folder.name
            call_title = self.get_call_title(call_id)
            callid_to_title[call_id] = call_title
            self.log(f"Processing call: {call_id} (title: {call_title})")
            channels = find_channels(call_folder)
            if not channels:
                self.log(f"No valid channels for {call_id}")
                continue
            utterances_by_channel = collect_utterances_merged(call_folder, channels)
            call_characters_dir = characters_root / call_title
            call_characters_dir.mkdir(parents=True, exist_ok=True)
            for channel, utts in utterances_by_channel.items():
                if not utts:
                    self.log(f"No utterances for {channel} in {call_id}")
                    continue
                transcript_lines = []
                for u in utts:
                    ts = f"{u['timestamp']:.2f}" if u['timestamp'] is not None else "?"
                    transcript_lines.append(f"[{ts}][{u['speaker_id']}] {u['text']}")
                transcript = '\n'.join(transcript_lines)
                out_dir = call_characters_dir / channel
                out_dir.mkdir(parents=True, exist_ok=True)
                transcript_path = out_dir / 'channel_transcript.txt'
                transcript_path.write_text(transcript, encoding='utf-8')
                # LLM persona generation
                user_prompt = (
                    f"Given the following transcript, generate a Character.AI persona {LENNYBOT_STYLE_NOTE}. Transcript:\n{transcript}"
                )
                persona_path = out_dir / 'persona.md'
                persona = run_llm_task(user_prompt, self.llm_config, output_path=persona_path, seed=None)
                audio_clips = create_audio_clips(utts, out_dir, channel)
                self.log(f"Wrote persona, transcript, and {len(audio_clips)} audio clips for {channel} in {call_title} to {out_dir}")
        self.log(f"Call ID to Title mapping: {json.dumps(callid_to_title, indent=2)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Character Persona Builder Extension")
    parser.add_argument('output_root', type=str, help='Root output folder (parent of speakers/)')
    parser.add_argument('--llm-config', type=str, default=None, help='Path to LLM config JSON (default: workflows/llm_tasks.json)')
    args = parser.parse_args()
    ext = CharacterPersonaBuilder(args.output_root, llm_config_path=args.llm_config)
    ext.run() 