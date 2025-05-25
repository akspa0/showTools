import os
import json
import shutil
from pathlib import Path
import re
from extension_base import ExtensionBase
from typing import List

# --- USAGE ---
"""
Usage:
    python character_ai_description_builder_v2.py <output_root>

This script processes finalized pipeline outputs to generate Character.AI persona files and audio clips for each speaker in each call.
Outputs are written to finalized/calls/<call_id>/character_ai_descriptions/SXX/.
"""

# Stub for LLM call (replace with actual LLM integration)
def generate_character_ai_persona(transcript: str, speaker_label: str) -> str:
    # Replace with actual LLM call
    return f"""{{
  "name": "{speaker_label}",
  "short_description": "Auto-generated persona for {speaker_label}",
  "long_description": "This persona is based on the transcript for {speaker_label}.",
  "greeting": "Hello!",
  "personality_traits": ["auto", "generated"],
  "example_dialog": [
    ["Hi!", "Hello!"],
    ["How are you?", "I'm an AI persona."]
  ]
}}"""

class CharacterAIDescriptionBuilder(ExtensionBase):
    def run(self):
        finalized = self.output_root / 'finalized'
        calls_dir = finalized / 'calls'
        speakers_dir = finalized / 'speakers'
        if not calls_dir.exists() or not speakers_dir.exists():
            self.log("Required directories not found.")
            return
        for call_folder in calls_dir.iterdir():
            if not call_folder.is_dir():
                continue
            call_id = call_folder.name
            self.log(f"Processing call: {call_id}")
            # Find all speakers with segments for this call
            for speaker_folder in speakers_dir.iterdir():
                if not speaker_folder.is_dir() or not speaker_folder.name.startswith('S'):
                    continue
                speaker_label = speaker_folder.name
                # Find all .txt files for this call in this speaker folder
                seg_txts = [f for f in speaker_folder.glob(f'{call_id}-*.txt')]
                if not seg_txts:
                    continue
                # Aggregate transcript
                transcript = '\n'.join([f.read_text(encoding='utf-8').strip() for f in sorted(seg_txts)])
                if not transcript.strip():
                    self.log(f"No transcript for {speaker_label} in {call_id}")
                    continue
                # Generate persona
                persona = generate_character_ai_persona(transcript, speaker_label)
                # Output folder
                out_dir = call_folder / 'character_ai_descriptions' / speaker_label
                out_dir.mkdir(parents=True, exist_ok=True)
                persona_path = out_dir / f'{speaker_label}_persona.json'
                persona_path.write_text(persona, encoding='utf-8')
                self.log(f"Wrote persona for {speaker_label} in {call_id} to {persona_path}")
                # Copy transcript
                transcript_path = out_dir / f'{speaker_label}_transcript.txt'
                transcript_path.write_text(transcript, encoding='utf-8')
                # Copy/collate audio segments for this speaker/call
                seg_wavs = [speaker_folder / f.with_suffix('.wav').name for f in seg_txts if (speaker_folder / f.with_suffix('.wav').name).exists()]
                if seg_wavs:
                    for wav_file in seg_wavs:
                        shutil.copy2(wav_file, out_dir / wav_file.name)
                else:
                    self.log(f"No audio segments for {speaker_label} in {call_id}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python character_ai_description_builder_v2.py <output_root>")
        exit(1)
    ext = CharacterAIDescriptionBuilder(sys.argv[1])
    ext.run() 