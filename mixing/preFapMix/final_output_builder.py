import os
import shutil
import json
import re
from pathlib import Path
from typing import List, Dict

# TODO: Import mutagen or similar for MP3 metadata
# TODO: Import ffmpeg or pydub for audio conversion

PROJECT_ROOT = Path(__file__).resolve().parent

class FinalOutputBuilder:
    def __init__(self, project_root: Path, output_dir: Path):
        self.project_root = project_root
        self.output_dir = output_dir
        self.calls_dir = output_dir / 'calls'
        self.soundbites_dir = output_dir / 'soundbites'
        self.llm_dir = output_dir / 'llm'
        self.show_dir = output_dir / 'show'
        self.metadata = {}

    def collect_calls(self):
        """
        Scan the processed_calls directory, gather all outputs and metadata for each call,
        and populate self.metadata['calls'] with a structured dictionary for each call.
        Log missing or irregular data for later [BAD] marking.
        """
        processed_calls_dir = self.project_root / 'processed_calls'
        self.metadata['calls'] = []
        if not processed_calls_dir.exists() or not processed_calls_dir.is_dir():
            print(f"[FinalOutputBuilder] processed_calls directory not found: {processed_calls_dir}")
            return
        for call_folder in processed_calls_dir.iterdir():
            if not call_folder.is_dir():
                continue
            call_data = {
                'call_id': call_folder.name,
                'audio_files': [],
                'transcripts': [],
                'llm_outputs': {},
                'soundbites': [],
                'metadata_jsons': [],
                'call_folder': str(call_folder.resolve()),
                'bad': False,
                'missing': []
            }
            # Main audio (WAV/MP3)
            for ext in ('.mp3', '.wav'):
                for audio_file in call_folder.glob(f'*{ext}'):
                    call_data['audio_files'].append(str(audio_file.resolve()))
            if not call_data['audio_files']:
                call_data['missing'].append('audio')
            # Transcripts
            for transcript_file in call_folder.glob('*.json'):
                call_data['transcripts'].append(str(transcript_file.resolve()))
            for transcript_file in call_folder.glob('*.txt'):
                if 'transcript' in transcript_file.name:
                    call_data['transcripts'].append(str(transcript_file.resolve()))
            if not call_data['transcripts']:
                call_data['missing'].append('transcript')
            # LLM outputs (name, synopsis, hashtags, etc.)
            for llm_file in call_folder.glob('*_suggested_name.txt'):
                call_data['llm_outputs']['name'] = str(llm_file.resolve())
            for llm_file in call_folder.glob('*_combined_call_summary.txt'):
                call_data['llm_outputs']['synopsis'] = str(llm_file.resolve())
            for llm_file in call_folder.glob('*_hashtags.txt'):
                call_data['llm_outputs']['hashtags'] = str(llm_file.resolve())
            # Any other LLM outputs
            for llm_file in call_folder.glob('*.txt'):
                if llm_file.name not in call_data['llm_outputs'].values():
                    call_data['llm_outputs'][llm_file.stem] = str(llm_file.resolve())
            # Soundbite directories
            for soundbite_dir in call_folder.iterdir():
                if soundbite_dir.is_dir() and (soundbite_dir.name.startswith('S') or soundbite_dir.name.startswith('RECV_') or soundbite_dir.name.startswith('TRANS_')):
                    for soundbite_file in soundbite_dir.glob('*.wav'):
                        call_data['soundbites'].append(str(soundbite_file.resolve()))
            if not call_data['soundbites']:
                call_data['missing'].append('soundbites')
            # Metadata JSONs
            for meta_file in call_folder.glob('*.json'):
                if meta_file.name not in [Path(t).name for t in call_data['transcripts']]:
                    call_data['metadata_jsons'].append(str(meta_file.resolve()))
            # Mark as bad if missing any critical data
            if call_data['missing']:
                call_data['bad'] = True
            self.metadata['calls'].append(call_data)
        print(f"[FinalOutputBuilder] Collected {len(self.metadata['calls'])} calls from {processed_calls_dir}")

    def convert_soundbites_to_mp3(self):
        """
        For each call, convert all .wav soundbites to .mp3 using ffmpeg (or pydub if available).
        Copy or enrich metadata (ID3 tags, JSON sidecar) for each MP3.
        Update the metadata structure to reference the new MP3 files.
        Optionally remove the original .wav after successful conversion.
        """
        import subprocess
        from mutagen.easyid3 import EasyID3
        from mutagen.id3 import ID3, TXXX, COMM
        for call in self.metadata.get('calls', []):
            mp3_soundbites = []
            for wav_path in call['soundbites']:
                wav_path_obj = Path(wav_path)
                mp3_path = wav_path_obj.with_suffix('.mp3')
                # Convert WAV to MP3 using ffmpeg
                ffmpeg_cmd = [
                    'ffmpeg', '-y', '-i', str(wav_path_obj),
                    '-codec:a', 'libmp3lame', '-qscale:a', '2', str(mp3_path)
                ]
                try:
                    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)
                    if result.returncode == 0 and mp3_path.exists():
                        # Copy/enrich metadata (ID3 tags, JSON sidecar)
                        try:
                            audio = EasyID3(str(mp3_path))
                        except Exception:
                            audio = ID3()
                        # Example: Add call_id and type as custom tags
                        audio['title'] = call['call_id']
                        # TODO: Add more tags (categories, lineage, etc.)
                        audio.save(str(mp3_path))
                        # Write JSON sidecar
                        sidecar_json = mp3_path.with_suffix('.mp3.json')
                        with open(sidecar_json, 'w') as f:
                            json.dump({
                                'source_wav': str(wav_path_obj),
                                'call_id': call['call_id'],
                                # TODO: Add more metadata as needed
                            }, f, indent=2)
                        mp3_soundbites.append(str(mp3_path.resolve()))
                        # Optionally remove original WAV
                        try:
                            wav_path_obj.unlink()
                        except Exception:
                            pass
                    else:
                        print(f"[FinalOutputBuilder] Failed to convert {wav_path_obj} to MP3: {result.stderr}")
                except Exception as e:
                    print(f"[FinalOutputBuilder] Exception during WAV to MP3 conversion: {e}")
            call['mp3_soundbites'] = mp3_soundbites
        print(f"[FinalOutputBuilder] Converted soundbites to MP3 for all calls.")
        # TODO: Content Moderation/Labeling:
        # - Optionally run Whisper with timestamps on the final show file and censor flagged words.
        # - Or, use LLM to evaluate each call's transcript for problematic content and label/flag accordingly.
        # - Use the first or second LLM-generated tag as a folder label for calls in the output.

    def organize_llm_responses(self):
        """
        For each call, parse all LLM outputs (name, synopsis, hashtags, and any additional LLM-generated files).
        Extract the main tags/categories for each call (from hashtags/categories file or similar).
        Prepare for folder labeling and content moderation by storing the first/second tag and all LLM outputs in the metadata.
        Update the metadata structure for easy downstream access.
        """
        for call in self.metadata.get('calls', []):
            llm_outputs = call.get('llm_outputs', {})
            # Parse name
            call['llm_name'] = None
            if 'name' in llm_outputs and Path(llm_outputs['name']).exists():
                with open(llm_outputs['name'], 'r', encoding='utf-8') as f:
                    call['llm_name'] = f.read().strip()
            # Parse synopsis
            call['llm_synopsis'] = None
            if 'synopsis' in llm_outputs and Path(llm_outputs['synopsis']).exists():
                with open(llm_outputs['synopsis'], 'r', encoding='utf-8') as f:
                    call['llm_synopsis'] = f.read().strip()
            # Parse hashtags/categories
            call['llm_tags'] = []
            if 'hashtags' in llm_outputs and Path(llm_outputs['hashtags']).exists():
                with open(llm_outputs['hashtags'], 'r', encoding='utf-8') as f:
                    tags_line = f.read().strip()
                    # Split by comma, space, or both
                    tags = [t.strip().lstrip('#') for t in re.split(r'[,#\s]+', tags_line) if t.strip()]
                    call['llm_tags'] = tags
            # Store first/second tag for folder labeling
            call['primary_tag'] = call['llm_tags'][0] if call['llm_tags'] else None
            call['secondary_tag'] = call['llm_tags'][1] if len(call['llm_tags']) > 1 else None
            # Store all other LLM outputs (any .txt not already parsed)
            call['llm_other_outputs'] = {}
            for key, path in llm_outputs.items():
                if key not in ['name', 'synopsis', 'hashtags'] and Path(path).exists():
                    with open(path, 'r', encoding='utf-8') as f:
                        call['llm_other_outputs'][key] = f.read().strip()
        print(f"[FinalOutputBuilder] Organized LLM responses for all calls.")

    def build_transcripts(self):
        """
        For each call, collect all transcript files (JSON and TXT), store their paths and optionally content in the metadata.
        Prepare a master transcript (aggregated across all calls, in order) and store its content/path in self.metadata['master_transcript'].
        Ensure transcript paths are ready for downstream steps.
        """
        master_transcript_lines = []
        for call in self.metadata.get('calls', []):
            call['transcript_json'] = None
            call['transcript_txt'] = None
            # Find JSON transcript
            for tpath in call.get('transcripts', []):
                if tpath.endswith('.json') and Path(tpath).exists():
                    call['transcript_json'] = tpath
            # Find TXT transcript
            for tpath in call.get('transcripts', []):
                if tpath.endswith('.txt') and Path(tpath).exists():
                    call['transcript_txt'] = tpath
            # Optionally, read transcript content for aggregation
            transcript_content = None
            if call['transcript_txt']:
                try:
                    with open(call['transcript_txt'], 'r', encoding='utf-8') as f:
                        transcript_content = f.read().strip()
                except Exception:
                    transcript_content = None
            elif call['transcript_json']:
                try:
                    with open(call['transcript_json'], 'r', encoding='utf-8') as f:
                        transcript_content = f.read().strip()
                except Exception:
                    transcript_content = None
            # Add to master transcript
            if transcript_content:
                master_transcript_lines.append(f"\n{'='*80}\nCALL: {call['call_id']}\n{'='*80}\n{transcript_content}\n")
        # Write master transcript to output_dir
        master_transcript_path = self.output_dir / 'master_transcript.txt'
        with open(master_transcript_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(master_transcript_lines))
        self.metadata['master_transcript'] = str(master_transcript_path.resolve())
        print(f"[FinalOutputBuilder] Built master transcript for all calls: {master_transcript_path}")

    def finalize_calls_with_tones(self, tone_path=None, tone_duration=1.0):
        """
        For each valid call, append a tone (default: 1s silence or beep, or a configurable file) to the end of the call MP3.
        Save the result as a new finalized MP3 (call_id_finalized.mp3).
        Update the call's metadata to reference the finalized MP3.
        """
        from pydub import AudioSegment
        for call in self.metadata.get('calls', []):
            if call.get('bad'):
                continue
            main_mp3 = None
            for afile in call.get('audio_files', []):
                if afile.endswith('.mp3') and Path(afile).exists():
                    main_mp3 = afile
                    break
            if not main_mp3:
                continue
            audio = AudioSegment.from_mp3(main_mp3)
            # Load or generate tone
            if tone_path and Path(tone_path).exists():
                tone = AudioSegment.from_file(tone_path)
            else:
                # Default: 1s silence (can be replaced with a beep or configurable file)
                tone = AudioSegment.silent(duration=int(tone_duration * 1000))
                # TODO: Optionally generate a beep or use a configurable tone file
            finalized_audio = audio + tone
            finalized_mp3_path = Path(main_mp3).with_name(f"{Path(main_mp3).stem}_finalized.mp3")
            finalized_audio.export(finalized_mp3_path, format="mp3")
            call['finalized_mp3'] = str(finalized_mp3_path.resolve())
        print(f"[FinalOutputBuilder] Finalized all calls with appended tones.")

    def build_show_file(self):
        """
        Concatenate all valid call finalized MP3s into a single show audio file (show_audio.mp3) using ffmpeg.
        Build a show transcript (show_transcript.txt) by concatenating per-call transcripts.
        Build a show timestamps file (show_timestamps.txt) with start times for each call in the show audio.
        Store all relevant paths in self.metadata for downstream use.
        """
        import subprocess
        from datetime import timedelta
        show_audio_path = self.output_dir / 'show_audio.mp3'
        show_transcript_path = self.output_dir / 'show_transcript.txt'
        show_timestamps_path = self.output_dir / 'show_timestamps.txt'
        # 1. Gather all valid call finalized MP3s (not BAD)
        call_mp3s = []
        call_ids = []
        for call in self.metadata.get('calls', []):
            if call.get('bad'):
                continue
            main_mp3 = call.get('finalized_mp3')
            if main_mp3 and Path(main_mp3).exists():
                call_mp3s.append(main_mp3)
                call_ids.append(call['call_id'])
        # 2. Concatenate MP3s using ffmpeg
        if call_mp3s:
            filelist_path = self.output_dir / 'show_audio_filelist.txt'
            with open(filelist_path, 'w', encoding='utf-8') as f:
                for mp3 in call_mp3s:
                    f.write(f"file '{mp3}'\n")
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', str(filelist_path), '-c', 'copy', str(show_audio_path)
            ]
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0 and show_audio_path.exists():
                print(f"[FinalOutputBuilder] Created show audio: {show_audio_path}")
            else:
                print(f"[FinalOutputBuilder] Failed to create show audio: {result.stderr}")
            try:
                filelist_path.unlink()
            except Exception:
                pass
        # 3. Build show transcript
        with open(show_transcript_path, 'w', encoding='utf-8') as f:
            for call in self.metadata.get('calls', []):
                if call.get('bad'):
                    continue
                if call.get('transcript_txt') and Path(call['transcript_txt']).exists():
                    with open(call['transcript_txt'], 'r', encoding='utf-8') as tf:
                        transcript_content = tf.read().strip()
                    f.write(f"\n{'='*80}\nCALL: {call['call_id']}\n{'='*80}\n{transcript_content}\n")
        # 4. Build show timestamps
        def get_audio_duration(audio_file):
            ffprobe_cmd = [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_file)
            ]
            try:
                process = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=False)
                if process.returncode == 0:
                    return float(process.stdout.strip())
            except Exception:
                pass
            return None
        with open(show_timestamps_path, 'w', encoding='utf-8') as f:
            f.write("# Show Timestamps - Format: HH:MM:SS - Call ID\n")
            current_time = 0.0
            for call, mp3 in zip(self.metadata.get('calls', []), call_mp3s):
                if call.get('bad'):
                    continue
                position_time = str(timedelta(seconds=int(current_time)))
                f.write(f"{position_time} - {call['call_id']}\n")
                duration = get_audio_duration(mp3)
                if duration:
                    current_time += duration
                else:
                    current_time += 300  # Fallback: 5 minutes
        # 5. Store paths in metadata
        self.metadata['show_audio'] = str(show_audio_path.resolve())
        self.metadata['show_transcript'] = str(show_transcript_path.resolve())
        self.metadata['show_timestamps'] = str(show_timestamps_path.resolve())
        print(f"[FinalOutputBuilder] Built show file, transcript, and timestamps.")

    def write_metadata(self):
        """
        Write a comprehensive metadata JSON file (final_output_metadata.json) in the output directory.
        Include all calls, soundbites, LLM outputs, transcripts, show file info, and any other relevant metadata.
        Ensure all paths are relative to the project root (use make_paths_relative if needed).
        """
        # Ensure all paths are relative to the project root
        self.make_paths_relative()
        metadata_path = self.output_dir / 'final_output_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        print(f"[FinalOutputBuilder] Wrote metadata to {metadata_path}")

    def mark_bad_calls(self):
        # TODO: Mark any call/segment as [BAD] if too short or missing data
        pass

    def make_paths_relative(self):
        # TODO: Ensure all paths in metadata are relative to the project root
        pass

    def zip_final_output(self):
        # TODO: Zip the entire final output folder and save in the root of the output directory
        pass

    def run(self):
        self.collect_calls()
        self.convert_soundbites_to_mp3()
        self.organize_llm_responses()
        self.build_transcripts()
        self.finalize_calls_with_tones()
        self.build_show_file()
        self.write_metadata()
        self.mark_bad_calls()
        self.make_paths_relative()
        self.zip_final_output()


def main():
    # TODO: Parse arguments or get paths from environment/config
    project_root = PROJECT_ROOT
    output_dir = project_root / 'final_output'
    builder = FinalOutputBuilder(project_root, output_dir)
    builder.run()

if __name__ == '__main__':
    main() 