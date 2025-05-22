import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Callable, Dict, Any
from tqdm import tqdm
from rich.console import Console
from rich.traceback import install as rich_traceback_install
import uuid
from file_ingestion import process_file_job, SUPPORTED_EXTENSIONS, TYPE_MAP
from audio_separation import separate_audio_file
from clap_annotation import annotate_clap_for_out_files
from speaker_diarization import batch_diarize, segment_speakers_from_diarization
from transcription import transcribe_segments
import torchaudio
import re
import soundfile as sf
from collections import defaultdict
# --- Finalization stage import ---
from finalization_stage import run_finalization_stage

rich_traceback_install()
console = Console()

# Sub-identifier mapping for tuple members
TUPLE_SUBID = {
    'left': 'a',      # recv_out
    'right': 'b',     # trans_out
    'out': 'c',       # out
}

class Job:
    def __init__(self, job_id: str, data: Dict[str, Any], job_type: str = 'rename'):
        self.job_id = job_id
        self.data = data  # Arbitrary metadata/state for the job
        self.state = {}
        self.success = True
        self.error = None
        self.progress = 0.0
        self.job_type = job_type  # 'rename' or 'separate'

class PipelineOrchestrator:
    def __init__(self, run_folder: Path):
        self.jobs: List[Job] = []
        self.log: List[Dict[str, Any]] = []
        self.manifest: List[Dict[str, Any]] = []
        self.run_folder = run_folder
        self._log_buffer: List[Dict[str, Any]] = []  # Buffer for log events
        self._console_buffer: List[str] = []         # Buffer for console output
        self._logging_enabled = False                # Only enable after PII is gone
    def add_job(self, job: Job):
        self.jobs.append(job)
    def log_event(self, level, event, details=None):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'event': event,
            'details': details or {}
        }
        self.log.append(entry)
        # Buffer console output until logging is enabled (after PII is gone)
        msg = None
        if level == 'ERROR':
            msg = f"[bold red]{level}[/] {event}: {details}"
        elif level == 'WARNING':
            msg = f"[yellow]{level}[/] {event}: {details}"
        else:
            msg = f"[green]{level}[/] {event}: {details}"
        if self._logging_enabled:
            console.print(msg)
        else:
            self._console_buffer.append(msg)
    def enable_logging(self):
        """
        Call this after all raw_inputs are deleted and outputs are anonymized.
        Flushes buffered logs to console.
        """
        self._logging_enabled = True
        for msg in self._console_buffer:
            console.print(msg)
        self._console_buffer.clear()
    def write_log(self):
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_path = self.run_folder / f'orchestrator-log-{ts}.json'
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.log, f, indent=2)
        if self._logging_enabled:
            console.print(f"[bold green]Orchestrator log written to {log_path}[/]")
        else:
            self._console_buffer.append(f"[bold green]Orchestrator log written to {log_path}[/]")
    def write_manifest(self):
        manifest_path = self.run_folder / 'manifest.json'
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(self.manifest, f, indent=2)
        if self._logging_enabled:
            console.print(f"[bold green]Manifest written to {manifest_path}[/]")
        else:
            self._console_buffer.append(f"[bold green]Manifest written to {manifest_path}[/]")
    def add_separation_jobs(self):
        """
        After renaming, create jobs for audio separation for each left/right file in renamed/.
        """
        renamed_dir = self.run_folder / 'renamed'
        for file in renamed_dir.iterdir():
            if file.is_file() and ('-left-' in file.name or '-right-' in file.name):
                job_data = {
                    'input_path': str(file),
                    'input_name': file.name
                }
                job_id = f"separate_{file.stem}"
                self.jobs.append(Job(job_id=job_id, data=job_data, job_type='separate'))

    def run_audio_separation_stage(self):
        """
        Run audio separation for all jobs of type 'separate'.
        Updates manifest and logs via orchestrator methods.
        """
        separated_dir = self.run_folder / 'separated'
        separated_dir.mkdir(exist_ok=True)
        model_path = 'mel_band_roformer_vocals_fv4_gabox.ckpt'
        separation_jobs = [job for job in self.jobs if job.job_type == 'separate']
        self.log_event('INFO', 'audio_separation_start', {'file_count': len(separation_jobs)})
        for job in separation_jobs:
            input_file = Path(job.data['input_path'])
            result = separate_audio_file(input_file, separated_dir, model_path)
            if result['separation_status'] == 'success':
                self.log_event('INFO', 'audio_separation_success', {
                    'input_name': result['input_name'],
                    'output_stems': [s['output_path'] for s in result['output_stems']]
                })
                job.success = True
            else:
                self.log_event('ERROR', 'audio_separation_failed', {
                    'input_name': result['input_name'],
                    'stderr': result['stderr']
                })
                job.success = False
                job.error = result['stderr']
            # Update manifest with separation results (anonymized)
            self.manifest.append({
                'stage': 'separated',
                'input_name': result['input_name'],
                'output_stems': result['output_stems'],
                'separation_status': result['separation_status']
            })
        # Optionally, write a separation manifest file
        manifest_path = separated_dir / 'separation_manifest.json'
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump([{
                'input_name': job.data['input_name'],
                'success': job.success,
                'error': job.error
            } for job in separation_jobs], f, indent=2)
        self.log_event('INFO', 'audio_separation_complete', {'manifest_path': str(manifest_path)})

    def run_clap_annotation_stage(self):
        """
        Run CLAP annotation for all 'out' files in renamed/.
        Updates manifest and logs via orchestrator methods.
        Uses the fused CLAP model for annotation.
        Logs every file written and updates manifest.
        """
        renamed_dir = self.run_folder / 'renamed'
        clap_dir = self.run_folder / 'clap'
        prompts = [
            'dog barking', 'DTMF', 'ringing', 'yelling', 'music', 'laughter', 'crying', 'doorbell', 'car horn', 'applause', 'gunshot', 'siren', 'footsteps', 'phone hangup', 'phone pickup', 'busy signal', 'static', 'noise', 'silence'
        ]
        self.log_event('INFO', 'clap_annotation_start', {'prompts': prompts})
        fused_clap_model = 'laion/clap-htsat-fused'
        results = annotate_clap_for_out_files(renamed_dir, clap_dir, prompts, model=fused_clap_model)
        for result in results:
            self.log_and_manifest(
                stage='clap',
                call_id=result.get('call_id'),
                input_files=[str(result.get('input_name'))],
                output_files=[str(result.get('annotation_path'))],
                params={'prompts': prompts, 'model': fused_clap_model},
                metadata={'accepted_annotations': result.get('accepted_annotations')},
                event='clap_annotation_result',
                result='success'
            )
            self.manifest.append({
                'stage': 'clap',
                'call_id': result['call_id'],
                'input_name': result['input_name'],
                'annotation_path': result['annotation_path'],
                'accepted_annotations': result['accepted_annotations']
            })
        self.log_event('INFO', 'clap_annotation_complete', {'count': len(results)})

    def run_diarization_stage(self, hf_token=None, min_speakers=None):
        """
        Run speaker diarization for all *-vocals.wav files in separated/<call id>/.
        Sets max_speakers=8 for all files (model limit).
        Updates manifest and logs via orchestrator methods.
        Logs every file written and updates manifest.
        """
        separated_dir = self.run_folder / 'separated'
        diarized_dir = self.run_folder / 'diarized'
        diarized_dir.mkdir(exist_ok=True)
        self.log_event('INFO', 'diarization_start', {'max_speakers': 8})
        results = batch_diarize(
            str(separated_dir),
            str(diarized_dir),
            hf_token=hf_token,
            min_speakers=min_speakers,
            max_speakers=8,
            progress=True
        )
        for result in results:
            self.log_and_manifest(
                stage='diarized',
                call_id=result.get('call_id'),
                input_files=[str(result.get('input_name'))],
                output_files=[str(result.get('json'))],
                params={'max_speakers': 8},
                metadata={'segments': len(result.get('segments', []))},
                event='diarization_result',
                result='success'
            )
            self.manifest.append({
                'stage': 'diarized',
                'call_id': result.get('call_id'),
                'input_name': result.get('input_name'),
                'rttm': result.get('rttm'),
                'json': result.get('json'),
                'segments': result.get('segments', [])
            })
        self.log_event('INFO', 'diarization_complete', {'count': len(results), 'max_speakers': 8})

    def run_speaker_segmentation_stage(self):
        """
        Segment each diarized *-vocals.wav file into per-speaker audio files, saving in speakers/<call id>/SXX/.
        Log segment metadata and update manifest.
        Logs every file written and updates manifest.
        """
        diarized_dir = self.run_folder / 'diarized'
        separated_dir = self.run_folder / 'separated'
        speakers_dir = self.run_folder / 'speakers'
        speakers_dir.mkdir(exist_ok=True)
        self.log_event('INFO', 'speaker_segmentation_start', {})
        results = segment_speakers_from_diarization(
            str(diarized_dir),
            str(separated_dir),
            str(speakers_dir),
            progress=True
        )
        for seg in results:
            self.log_and_manifest(
                stage='speaker_segmented',
                call_id=seg.get('call_id'),
                input_files=[str(seg.get('input_wav'))],
                output_files=[str(seg.get('wav'))],
                params={'channel': seg.get('channel'), 'speaker': seg.get('speaker')},
                metadata={'start': seg.get('start'), 'end': seg.get('end')},
                event='speaker_segment',
                result='success'
            )
            self.manifest.append({
                'stage': 'speaker_segmented',
                **seg
            })
        self.log_event('INFO', 'speaker_segmentation_complete', {'count': len(results)})

    def run_resample_segments_stage(self):
        """
        Resample all speaker segment WAVs to 16kHz mono for ASR. Save as <segment>_16k.wav and update manifest.
        Overwrites any old _16k.wav files. Adds debug logging for waveform shape.
        Logs every file written and updates manifest.
        """
        speakers_dir = self.run_folder / 'speakers'
        resampled_segments = []
        self.log_event('INFO', 'resample_segments_start', {})
        for entry in self.manifest:
            if entry.get('stage') == 'speaker_segmented' and 'wav' in entry:
                wav_path = Path(entry['wav'])
                if not wav_path.exists():
                    continue
                out_path = wav_path.with_name(wav_path.stem + '_16k.wav')
                if out_path.exists():
                    out_path.unlink()
                waveform, sr = torchaudio.load(str(wav_path))
                print(f"[DEBUG] Loaded: {wav_path} shape={waveform.shape} sr={sr} dtype={waveform.dtype}")
                if waveform.ndim == 3:
                    if waveform.shape[0] == 1 and waveform.shape[2] == 2:
                        waveform = waveform.squeeze(0).mean(dim=1, keepdim=True).transpose(0, 1)
                        print(f"[DEBUG] Squeezed and averaged: shape={waveform.shape}")
                    elif waveform.shape[0] == 1 and waveform.shape[1] == 2:
                        waveform = waveform.squeeze(0).mean(dim=0, keepdim=True)
                        print(f"[DEBUG] Squeezed and averaged: shape={waveform.shape}")
                    else:
                        print(f"[WARN] Unexpected 3D shape: {waveform.shape}")
                elif waveform.ndim == 2:
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                        print(f"[DEBUG] Averaged channels: shape={waveform.shape}")
                else:
                    print(f"[WARN] Unexpected waveform ndim: {waveform.ndim} shape={waveform.shape}")
                if sr != 16000:
                    waveform = torchaudio.functional.resample(waveform, sr, 16000)
                    sr = 16000
                    print(f"[DEBUG] After resample: shape={waveform.shape} sr={sr}")
                if waveform.ndim != 2 or waveform.shape[0] != 1:
                    print(f"[WARN] Skipping {wav_path}: shape after mono/resample is {waveform.shape}, expected [1, time]")
                    continue
                torchaudio.save(str(out_path), waveform, sr)
                self.log_and_manifest(
                    stage='resampled',
                    call_id=entry.get('call_id'),
                    input_files=[str(wav_path)],
                    output_files=[str(out_path)],
                    params={'target_sr': 16000},
                    metadata=None,
                    event='file_written',
                    result='success'
                )
                resampled_entry = dict(entry)
                resampled_entry['wav_16k'] = str(out_path)
                resampled_entry['stage'] = 'resampled'
                self.manifest.append(resampled_entry)
                resampled_segments.append(resampled_entry)
        self.log_event('INFO', 'resample_segments_complete', {'count': len(resampled_segments)})

    def run_transcription_stage(self, asr_engine='parakeet', asr_config=None):
        """
        Transcribe all speaker segments using the selected ASR engine (parakeet or whisper).
        Updates manifest with transcript results.
        Logs every file written and updates manifest.
        """
        speakers_dir = self.run_folder / 'speakers'
        segments = [entry for entry in self.manifest if entry.get('stage') == 'resampled' and 'wav_16k' in entry]
        if not segments:
            segments = []
            for call_id in os.listdir(speakers_dir):
                call_dir = speakers_dir / call_id
                if not call_dir.is_dir():
                    continue
                for channel in os.listdir(call_dir):
                    channel_dir = call_dir / channel
                    if not channel_dir.is_dir():
                        continue
                    for spk in os.listdir(channel_dir):
                        spk_dir = channel_dir / spk
                        if not spk_dir.is_dir():
                            continue
                        for wav in os.listdir(spk_dir):
                            if wav.endswith('_16k.wav'):
                                wav_path = spk_dir / wav
                                parts = wav.replace('_16k','')[:-4].split('-')
                                index = int(parts[0]) if parts and parts[0].isdigit() else None
                                start = float(parts[1])/100 if len(parts) > 1 else None
                                end = float(parts[2])/100 if len(parts) > 2 else None
                                segments.append({
                                    'call_id': call_id,
                                    'channel': channel,
                                    'speaker': spk,
                                    'index': index,
                                    'start': start,
                                    'end': end,
                                    'wav': str(wav_path)
                                })
        if asr_config is None:
            asr_config = {}
        asr_config = {**asr_config, 'asr_engine': asr_engine}
        results = transcribe_segments(segments, asr_config)
        for res in results:
            self.log_and_manifest(
                stage='transcribed',
                call_id=res.get('call_id'),
                input_files=[str(res.get('wav'))],
                output_files=[str(res.get('txt')), str(res.get('json'))] if res.get('txt') and res.get('json') else [],
                params={'asr_engine': asr_engine},
                metadata={'error': res.get('error') if 'error' in res else None},
                event='transcription',
                result='success' if not res.get('error') else 'error',
                error=res.get('error') if 'error' in res else None
            )
            self.manifest.append({
                **res,
                'stage': 'transcribed'
            })
        self.log_event('INFO', 'transcription_complete', {'count': len(results)})

    @staticmethod
    def sanitize(text, max_words=12, max_length=40):
        if not text:
            return 'untitled'
        words = re.findall(r'\w+', text)[:max_words]
        short = '_'.join(words)
        return short[:max_length] or 'untitled'

    def run_rename_soundbites_stage(self):
        """
        No longer renames files in speakers/ folder. Renaming and copying now happens in run_final_soundbite_stage.
        This stage is now a no-op for file renaming, but still updates manifest for valid soundbites.
        Skips soundbites with missing/empty transcript or duration < 1s.
        Logs every file written and updates manifest.
        """
        valid_count = 0
        for entry in self.manifest:
            if entry.get('stage') == 'transcribed' and 'wav' in entry and 'text' in entry:
                transcript = entry['text']
                if not transcript or not transcript.strip():
                    continue
                wav_path = Path(entry['wav'])
                try:
                    info = sf.info(str(wav_path))
                    duration = info.duration
                except Exception:
                    duration = None
                if duration is not None and duration < 1.0:
                    continue
                entry['stage'] = 'soundbite_valid'
                valid_count += 1
                self.log_and_manifest(
                    stage='soundbite_valid',
                    call_id=entry.get('call_id'),
                    input_files=[str(entry.get('wav'))],
                    output_files=[str(entry.get('wav'))],
                    params=None,
                    metadata={'duration': duration},
                    event='soundbite_valid',
                    result='success'
                )
        self.log_event('INFO', 'soundbite_valid_count', {'count': valid_count})

    def run_final_soundbite_stage(self):
        """
        Copy and rename valid soundbites from speakers/ to soundbites/ folder, using <index>-<short_transcription>.* naming.
        Only includes valid soundbites (with transcript and duration >= 1s).
        Formats master transcript as [CHANNEL][SpeakerXX][start-end]: Transcription.
        Also generates new segment logs in soundbites/<call_id>/<channel>/ with transcript text.
        The master transcript for each call is the canonical input for LLM tasks.
        Integrates CLAP events with confidence >= 0.90 as [CLAP][start-end]: <label> (high confidence), sorted chronologically.
        Logs every file written and updates manifest.
        Filters out malformed segment entries.
        """
        import shutil
        import json
        speakers_dir = self.run_folder / 'speakers'
        separated_dir = self.run_folder / 'separated'
        soundbites_dir = self.run_folder / 'soundbites'
        soundbites_dir.mkdir(exist_ok=True)
        from collections import defaultdict
        segments = [entry for entry in self.manifest if entry.get('stage') == 'speaker_segmented']
        calls = defaultdict(list)
        for seg in segments:
            calls[seg.get('call_id')].append(seg)
        for call_id, segs in calls.items():
            call_soundbites = []
            channel_segments = defaultdict(list)
            clap_events = []
            clap_dir = self.run_folder / 'clap' / call_id
            for channel in ['left-vocals', 'right-vocals']:
                clap_json = clap_dir / f"{channel}_clap_annotations.json"
                if clap_json.exists():
                    with open(clap_json, 'r', encoding='utf-8') as f:
                        clap_data = json.load(f)
                    for event in clap_data.get('events', []):
                        if event.get('confidence', 0) >= 0.90:
                            clap_events.append({
                                'start': event.get('start', 0),
                                'end': event.get('end', 0),
                                'label': event.get('label', 'unknown'),
                                'confidence': event.get('confidence'),
                                'channel': channel
                            })
            # Filter segs to only those with 'channel' and 'start', log malformed
            valid_segs = []
            for s in segs:
                if 'channel' in s and 'start' in s:
                    valid_segs.append(s)
                else:
                    self.log_event('WARNING', 'malformed_segment_entry', {'entry': s})
            for seg in sorted(valid_segs, key=lambda s: (s['channel'], s['start'])):
                channel = seg['channel']
                speaker = seg['speaker']
                start = seg['start']
                end = seg['end']
                duration = end - start if end and start else 0
                if duration < 1.0:
                    continue
                idx = seg['index']
                transcript_entry = next((e for e in self.manifest if e.get('stage') == 'soundbite_valid' and e.get('call_id') == call_id and e.get('channel') == channel and e.get('speaker') == speaker and e.get('index') == idx), None)
                transcript = transcript_entry['text'] if transcript_entry and transcript_entry.get('text') else None
                if not transcript or not transcript.strip():
                    continue
                channel_fmt = f"[{channel.replace('-vocals','').upper()}]"
                spk_num = ''.join(filter(str.isdigit, speaker))
                spk_fmt = f"[Speaker{int(spk_num):02d}]" if spk_num else f"[{speaker}]"
                spk_dir = speakers_dir / call_id / channel / speaker
                base_index = f"{idx:04d}"
                orig_wav = next((spk_dir / f for f in os.listdir(spk_dir) if f.startswith(base_index) and f.endswith('.wav')), None)
                orig_txt = next((spk_dir / f for f in os.listdir(spk_dir) if f.startswith(base_index) and f.endswith('.txt')), None)
                orig_json = next((spk_dir / f for f in os.listdir(spk_dir) if f.startswith(base_index) and f.endswith('.json')), None)
                out_dir = soundbites_dir / call_id / channel / speaker
                out_dir.mkdir(parents=True, exist_ok=True)
                out_base = f"{idx:04d}-{self.sanitize(transcript)}"
                out_wav = out_dir / (out_base + '.wav')
                out_txt = out_dir / (out_base + '.txt') if orig_txt else None
                out_json = out_dir / (out_base + '.json') if orig_json else None
                if orig_wav and os.path.exists(orig_wav):
                    shutil.copy2(orig_wav, out_wav)
                if orig_txt and os.path.exists(orig_txt) and out_txt:
                    shutil.copy2(orig_txt, out_txt)
                if orig_json and os.path.exists(orig_json) and out_json:
                    shutil.copy2(orig_json, out_json)
                self.log_and_manifest(
                    stage='final_soundbite',
                    call_id=call_id,
                    input_files=[str(orig_wav), str(orig_txt) if orig_txt else None, str(orig_json) if orig_json else None],
                    output_files=[str(out_wav), str(out_txt) if out_txt else None, str(out_json) if out_json else None],
                    params={'channel': channel, 'speaker': speaker},
                    metadata={'duration': duration},
                    event='file_written',
                    result='success'
                )
                manifest_entry = dict(seg)
                manifest_entry['stage'] = 'final_soundbite'
                manifest_entry['soundbite_wav'] = str(out_wav)
                manifest_entry['transcript'] = transcript
                manifest_entry['txt'] = str(out_txt) if out_txt else None
                manifest_entry['json'] = str(out_json) if out_json else None
                self.manifest.append(manifest_entry)
                call_soundbites.append({
                    'channel_fmt': channel_fmt,
                    'spk_fmt': spk_fmt,
                    'start': start,
                    'end': end,
                    'transcript': transcript
                })
                channel_segments[channel].append({
                    'speaker': speaker,
                    'start': start,
                    'end': end,
                    'transcript': transcript
                })
            all_events = [
                {
                    'type': 'soundbite',
                    'start': s['start'],
                    'end': s['end'],
                    'text': f"{s['channel_fmt']}{s['spk_fmt']}[{s['start']:.2f}-{s['end']:.2f}]: {s['transcript']}"
                } for s in call_soundbites
            ] + [
                {
                    'type': 'clap',
                    'start': e['start'],
                    'end': e['end'],
                    'text': f"[CLAP][{e['start']:.2f}-{e['end']:.2f}]: {e['label']} (high confidence)"
                } for e in clap_events
            ]
            all_events_sorted = sorted(all_events, key=lambda x: x['start'])
            master_txt = soundbites_dir / call_id / f"{call_id}_master_transcript.txt"
            with open(master_txt, 'w', encoding='utf-8') as f:
                for ev in all_events_sorted:
                    f.write(f"{ev['text']}\n")
            self.log_and_manifest(
                stage='master_transcript',
                call_id=call_id,
                input_files=None,
                output_files=[str(master_txt)],
                params=None,
                metadata=None,
                event='file_written',
                result='success'
            )
            master_json = soundbites_dir / call_id / f"{call_id}_master_transcript.json"
            with open(master_json, 'w', encoding='utf-8') as f:
                json.dump(all_events_sorted, f, indent=2, ensure_ascii=False)
            self.log_and_manifest(
                stage='master_transcript',
                call_id=call_id,
                input_files=None,
                output_files=[str(master_json)],
                params=None,
                metadata=None,
                event='file_written',
                result='success'
            )
            for channel, segs in channel_segments.items():
                seg_dir = soundbites_dir / call_id / channel
                seg_dir.mkdir(parents=True, exist_ok=True)
                seg_txt = seg_dir / f"{channel}_segments.txt"
                seg_json = seg_dir / f"{channel}_segments.json"
                with open(seg_txt, 'w', encoding='utf-8') as f:
                    for s in segs:
                        spk_num = ''.join(filter(str.isdigit, s['speaker']))
                        spk_fmt = f"[Speaker{int(spk_num):02d}]" if spk_num else f"[{s['speaker']}]"
                        f.write(f"{spk_fmt}[{s['start']:.2f}-{s['end']:.2f}] {s['transcript']}\n")
                self.log_and_manifest(
                    stage='segment_log',
                    call_id=call_id,
                    input_files=None,
                    output_files=[str(seg_txt)],
                    params={'channel': channel},
                    metadata=None,
                    event='file_written',
                    result='success'
                )
                with open(seg_json, 'w', encoding='utf-8') as f:
                    json.dump(segs, f, indent=2, ensure_ascii=False)
                self.log_and_manifest(
                    stage='segment_log',
                    call_id=call_id,
                    input_files=None,
                    output_files=[str(seg_json)],
                    params={'channel': channel},
                    metadata=None,
                    event='file_written',
                    result='success'
                )
        self.log_event('INFO', 'final_soundbites_complete', {'calls': list(calls.keys())})

    def get_master_transcript_path(self, call_id):
        """
        Returns the path to the canonical master transcript for a given call, suitable for LLM input.
        """
        soundbites_dir = self.run_folder / 'soundbites'
        return soundbites_dir / call_id / f"{call_id}_master_transcript.txt"

    def run_llm_task_for_call(self, call_id, master_transcript, llm_config, output_dir, llm_tasks):
        """
        For a given call, run all LLM tasks using the master transcript and config.
        Sends each prompt to the LLM API and writes output to output_dir.
        Returns a dict of output file paths.
        """
        import requests
        import json
        output_paths = {}
        with open(master_transcript, 'r', encoding='utf-8') as f:
            transcript = f.read()
        base_url = llm_config.get('lm_studio_base_url', 'http://localhost:1234/v1')
        api_key = llm_config.get('lm_studio_api_key', 'lm-studio')
        model_id = llm_config.get('lm_studio_model_identifier', 'llama-3.1-8b-supernova-etherealhermes')
        temperature = llm_config.get('lm_studio_temperature', 0.5)
        max_tokens = llm_config.get('lm_studio_max_tokens', 250)
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        for task in llm_tasks:
            name = task.get('name', 'unnamed_task')
            prompt_template = task.get('prompt_template', '')
            output_file = task.get('output_file', f'{name}.txt')
            prompt = prompt_template.format(transcript=transcript)
            data = {
                "model": model_id,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            out_path = output_dir / output_file
            try:
                response = requests.post(f"{base_url}/chat/completions", headers=headers, data=json.dumps(data), timeout=60)
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content'].strip()
                    with open(out_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    output_paths[name] = str(out_path)
                else:
                    error_msg = f"LLM API error {response.status_code}: {response.text}"
                    with open(out_path, 'w', encoding='utf-8') as f:
                        f.write(error_msg)
                    output_paths[name] = str(out_path)
                    self.log_event('ERROR', 'llm_api_error', {'call_id': call_id, 'task': name, 'status': response.status_code, 'text': response.text})
            except Exception as e:
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(f"LLM request failed: {e}")
                output_paths[name] = str(out_path)
                self.log_event('ERROR', 'llm_request_failed', {'call_id': call_id, 'task': name, 'error': str(e)})
        return output_paths

    def run_llm_stage(self, llm_config_path=None):
        """
        Run LLM tasks for each call using the master transcript as input.
        LLM tasks are defined in workflows/llm_tasks.json by default, or can be overridden via llm_config_path.
        For each call, for each LLM task, the response is saved to output/<run_id>/llm/<call_id>/<output_file>.
        Manifest is updated with LLM output paths and prompt lineage.
        Logs every file written and updates manifest.
        """
        import json
        from pathlib import Path
        if llm_config_path is None:
            llm_config_path = Path('workflows/llm_tasks.json')
        else:
            llm_config_path = Path(llm_config_path)
        if not llm_config_path.exists():
            self.log_event('ERROR', 'llm_config_missing', {'path': str(llm_config_path)})
            return
        with open(llm_config_path, 'r', encoding='utf-8') as f:
            llm_config = json.load(f)
        llm_tasks = llm_config.get('llm_tasks', [])
        soundbites_dir = self.run_folder / 'soundbites'
        llm_dir = self.run_folder / 'llm'
        llm_dir.mkdir(exist_ok=True)
        call_ids = [d.name for d in soundbites_dir.iterdir() if d.is_dir()]
        self.log_event('INFO', 'llm_stage_start', {'llm_dir': str(llm_dir)})
        for call_id in call_ids:
            master_transcript = self.get_master_transcript_path(call_id)
            if not master_transcript.exists():
                self.log_event('WARNING', 'master_transcript_missing', {'call_id': call_id})
                continue
            call_llm_dir = llm_dir / call_id
            call_llm_dir.mkdir(parents=True, exist_ok=True)
            output_paths = self.run_llm_task_for_call(
                call_id=call_id,
                master_transcript=master_transcript,
                llm_config=llm_config,
                output_dir=call_llm_dir,
                llm_tasks=llm_tasks
            )
            for task in llm_tasks:
                name = task.get('name', 'unnamed_task')
                output_file = task.get('output_file', f'{name}.txt')
                output_path = call_llm_dir / output_file
                self.log_and_manifest(
                    stage='llm',
                    call_id=call_id,
                    input_files=[str(master_transcript)],
                    output_files=[str(output_path)],
                    params={'task': name, 'prompt_template': task.get('prompt_template')},
                    metadata={'llm_model': llm_config.get('lm_studio_model_identifier')},
                    event='llm_task',
                    result='success'
                )
            self.log_event('INFO', 'llm_tasks_complete', {'call_id': call_id, 'llm_outputs': output_paths})

    def log_and_manifest(self, stage, call_id=None, input_files=None, output_files=None, params=None, metadata=None, event='file_written', result='success', error=None):
        """
        Helper to log and add manifest entry for any file operation.
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'stage': stage,
            'call_id': call_id,
            'input_files': input_files,
            'output_files': output_files,
            'params': params,
            'metadata': metadata,
            'result': result,
            'error': error
        }
        self.log_event('INFO' if result == 'success' else 'ERROR', event, entry)
        manifest_entry = {
            'stage': stage,
            'call_id': call_id,
            'input_files': input_files,
            'output_files': output_files,
            'params': params,
            'metadata': metadata,
            'result': result,
            'error': error
        }
        self.manifest.append(manifest_entry)

    def run_normalization_stage(self):
        """
        Normalize separated vocal stems to -14.0 LUFS and output to normalized/<call_id>/<channel>.wav.
        Downstream stages use normalized vocals.
        Logs every file written and updates manifest.
        """
        import pyloudnorm as pyln
        import soundfile as sf
        from pathlib import Path
        separated_dir = self.run_folder / 'separated'
        normalized_dir = self.run_folder / 'normalized'
        normalized_dir.mkdir(exist_ok=True)
        meter = pyln.Meter(44100)
        self.log_event('INFO', 'normalization_start', {'dir': str(separated_dir)})
        for call_id in os.listdir(separated_dir):
            call_sep_dir = separated_dir / call_id
            if not call_sep_dir.is_dir():
                continue
            call_norm_dir = normalized_dir / call_id
            call_norm_dir.mkdir(parents=True, exist_ok=True)
            for channel in ['left-vocals', 'right-vocals']:
                src = call_sep_dir / f"{channel}.wav"
                if not src.exists():
                    continue
                audio, sr = sf.read(str(src))
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != 44100:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=44100)
                    sr = 44100
                loudness = meter.integrated_loudness(audio)
                audio_norm = pyln.normalize.loudness(audio, loudness, -14.0)
                out_path = call_norm_dir / f"{channel}.wav"
                sf.write(str(out_path), audio_norm, sr)
                self.log_and_manifest(
                    stage='normalized',
                    call_id=call_id,
                    input_files=[str(src)],
                    output_files=[str(out_path)],
                    params={'target_lufs': -14.0},
                    metadata={'measured_lufs': loudness},
                    event='file_written',
                    result='success'
                )
        self.log_event('INFO', 'normalization_complete', {'normalized_dir': str(normalized_dir)})

    def run_remix_stage(self, call_tones=False):
        """
        For each call, mix normalized vocals + 50% instrumental for each channel, then combine with 60/40 stereo panning:
        stereo_left = 0.6 * left_mix + 0.4 * right_mix
        stereo_right = 0.4 * left_mix + 0.6 * right_mix
        Output stereo remixed_call.wav. If call_tones is set, append tones.wav from project root to the end.
        Logs every file written and updates manifest.
        Always uses normalized vocals from normalized/ folder.
        """
        import soundfile as sf
        import numpy as np
        from pathlib import Path
        normalized_dir = self.run_folder / 'normalized'
        separated_dir = self.run_folder / 'separated'
        call_dir = self.run_folder / 'call'
        call_dir.mkdir(exist_ok=True)
        tones_path = Path('tones.wav')
        self.log_event('INFO', 'remix_start', {'normalized_dir': str(normalized_dir)})
        for call_id in os.listdir(normalized_dir):
            call_norm_dir = normalized_dir / call_id
            call_sep_dir = separated_dir / call_id
            if not call_norm_dir.is_dir() or not call_sep_dir.is_dir():
                continue
            out_call_dir = call_dir / call_id
            out_call_dir.mkdir(parents=True, exist_ok=True)
            # Always use normalized vocals for remix
            left_vocal_path = call_norm_dir / "left-vocals.wav"
            right_vocal_path = call_norm_dir / "right-vocals.wav"
            left_instr_path = call_sep_dir / "left-instrumental.wav"
            right_instr_path = call_sep_dir / "right-instrumental.wav"
            if not (left_vocal_path.exists() and right_vocal_path.exists() and left_instr_path.exists() and right_instr_path.exists()):
                continue
            # Mix vocals + 0.5 * instrumental for each channel
            left_vocal, sr = sf.read(str(left_vocal_path))
            right_vocal, _ = sf.read(str(right_vocal_path))
            left_instr, _ = sf.read(str(left_instr_path))
            right_instr, _ = sf.read(str(right_instr_path))
            if left_vocal.ndim > 1:
                left_vocal = left_vocal.mean(axis=1)
            if right_vocal.ndim > 1:
                right_vocal = right_vocal.mean(axis=1)
            if left_instr.ndim > 1:
                left_instr = left_instr.mean(axis=1)
            if right_instr.ndim > 1:
                right_instr = right_instr.mean(axis=1)
            left_mix = left_vocal + 0.5 * left_instr
            right_mix = right_vocal + 0.5 * right_instr
            # 60/40 panning: stereo_left = 0.6*left_mix + 0.4*right_mix, stereo_right = 0.4*left_mix + 0.6*right_mix
            min_len = min(len(left_mix), len(right_mix))
            left_mix = left_mix[:min_len]
            right_mix = right_mix[:min_len]
            stereo_left = 0.6 * left_mix + 0.4 * right_mix
            stereo_right = 0.4 * left_mix + 0.6 * right_mix
            stereo = np.stack([stereo_left, stereo_right], axis=-1)
            remixed_path = out_call_dir / "remixed_call.wav"
            sf.write(str(remixed_path), stereo, sr)
            self.log_and_manifest(
                stage='remix',
                call_id=call_id,
                input_files=[str(left_vocal_path), str(right_vocal_path), str(left_instr_path), str(right_instr_path)],
                output_files=[str(remixed_path)],
                params={'panning': '60/40 split'},
                metadata=None,
                event='file_written',
                result='success'
            )
            if call_tones and tones_path.exists():
                remixed, _ = sf.read(str(remixed_path))
                tones, tones_sr = sf.read(str(tones_path))
                if tones_sr != sr:
                    import librosa
                    tones = librosa.resample(tones, orig_sr=tones_sr, target_sr=sr)
                remixed_full = np.concatenate([remixed, tones], axis=0)
                sf.write(str(remixed_path), remixed_full, sr)
                self.log_and_manifest(
                    stage='remix',
                    call_id=call_id,
                    input_files=[str(remixed_path), str(tones_path)],
                    output_files=[str(remixed_path)],
                    params={'append_tones': True},
                    metadata=None,
                    event='file_written',
                    result='success'
                )
        self.log_event('INFO', 'remix_complete', {'call_dir': str(call_dir)})

    def run_show_stage(self, call_tones=False):
        """
        Concatenate all successful remixed calls (with tones if needed) into show/show.wav (44.1kHz, stereo, 16-bit).
        Write show/show.json with start/end times for each call and tones, plus call metadata.
        Panning is preserved from the remix stage for each call.
        Logs every file written and updates manifest.
        Always uses remixed calls built from normalized vocals.
        """
        import soundfile as sf
        import numpy as np
        from pathlib import Path
        show_dir = self.run_folder / 'show'
        show_dir.mkdir(exist_ok=True)
        call_dir = self.run_folder / 'call'
        tones_path = Path('tones.wav')
        show_wav_path = show_dir / 'show.wav'
        show_json_path = show_dir / 'show.json'
        call_files = []
        self.log_event('INFO', 'show_start', {'call_dir': str(call_dir)})
        for call_id in sorted(os.listdir(call_dir)):
            remixed_path = call_dir / call_id / 'remixed_call.wav'
            if remixed_path.exists():
                call_files.append((call_id, remixed_path))
        show_audio = []
        show_timeline = []
        sr = 44100
        cur_time = 0.0
        llm_dir = self.run_folder / 'llm'
        for idx, (call_id, call_path) in enumerate(call_files):
            audio, file_sr = sf.read(str(call_path))
            if file_sr != sr:
                import librosa
                audio = librosa.resample(audio.T, orig_sr=file_sr, target_sr=sr).T
            start = cur_time
            end = start + audio.shape[0] / sr
            show_audio.append(audio)
            # --- Add LLM call title ---
            call_title = call_id
            call_title_path = llm_dir / call_id / 'call_title.txt'
            if call_title_path.exists():
                with open(call_title_path, 'r', encoding='utf-8') as f:
                    title = f.read().strip().strip('"')
                    if title:
                        call_title = title
            show_timeline.append({
                'call_id': call_id,
                'call_title': call_title,
                'start': start,
                'end': end
            })
            cur_time = end
            # Tones are already appended to each call in the remix stage; do NOT insert tones between calls here.
        if show_audio:
            show_audio = np.concatenate(show_audio, axis=0)
            sf.write(str(show_wav_path), show_audio, sr, subtype='PCM_16')
            self.log_and_manifest(
                stage='show',
                call_id=None,
                input_files=[str(f[1]) for f in call_files],
                output_files=[str(show_wav_path)],
                params={'call_tones': call_tones},
                metadata={'timeline': show_timeline},
                event='file_written',
                result='success'
            )
            with open(show_json_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(show_timeline, f, indent=2)
            self.log_and_manifest(
                stage='show',
                call_id=None,
                input_files=None,
                output_files=[str(show_json_path)],
                params=None,
                metadata=None,
                event='file_written',
                result='success'
            )
        self.log_event('INFO', 'show_complete', {'show_wav': str(show_wav_path), 'show_json': str(show_json_path)})
        # --- Finalization stage ---
        run_finalization_stage(self.run_folder, self.manifest)

    def run(self, call_tones=False):
        with tqdm(total=len(self.jobs), desc="Global Progress", position=0) as global_bar:
            for job in self.jobs:
                if job.job_type == 'rename':
                    result = process_file_job(job, self.run_folder)
                    manifest_entry = {
                        'job_id': job.job_id,
                        'tuple_index': job.data.get('tuple_index'),
                        'subid': job.data.get('subid'),
                        'type': job.data.get('file_type'),
                        'timestamp': job.data.get('timestamp'),
                        'output_name': result.get('output_name'),
                        'output_path': result.get('output_path'),
                        'stage': 'renamed',
                    }
                    if result['success']:
                        self.log_event('INFO', 'file_renamed', {
                            'output_name': result.get('output_name'),
                            'output_path': result.get('output_path')
                        })
                        self.manifest.append(manifest_entry)
                    else:
                        self.log_event('ERROR', 'file_rename_failed', {
                            'output_name': result.get('output_name'),
                            'output_path': result.get('output_path'),
                            'error': result.get('error')
                        })
                        self.manifest.append(manifest_entry)
                    global_bar.update(1)
        self.enable_logging()
        self.write_log()
        self.write_manifest()
        self.add_separation_jobs()
        self.run_audio_separation_stage()
        self.run_normalization_stage()
        self.run_clap_annotation_stage()
        self.run_diarization_stage()
        self.run_speaker_segmentation_stage()
        self.run_resample_segments_stage()
        self.run_transcription_stage(asr_engine='parakeet')
        self.run_rename_soundbites_stage()
        self.run_final_soundbite_stage()
        self.run_llm_stage()
        self.run_remix_stage(call_tones=call_tones)
        self.run_show_stage(call_tones=call_tones)

def create_jobs_from_input(input_dir: Path) -> List[Job]:
    all_files = []
    seen_names = set()
    for root, _, files in os.walk(input_dir):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            orig_path = Path(root) / file
            base_name = Path(file).name
            if base_name in seen_names:
                base_name = f"{Path(file).stem}_{uuid.uuid4().hex[:6]}{Path(file).suffix}"
            seen_names.add(base_name)
            all_files.append((orig_path, base_name))
    # Group files into tuples by timestamp
    tuple_groups = {}
    singles = []
    for orig_path, base_name in all_files:
        file_type, _ = extract_type(base_name)
        ts = os.path.getmtime(orig_path)
        ts_str = datetime.fromtimestamp(ts).strftime('%Y%m%d-%H%M%S')
        ext = Path(base_name).suffix.lower()
        if file_type:
            tuple_key = ts_str  # Only timestamp for grouping
            if tuple_key not in tuple_groups:
                tuple_groups[tuple_key] = []
            tuple_groups[tuple_key].append((orig_path, base_name, file_type, ts_str, ext))
        else:
            singles.append((orig_path, base_name, ts_str, ext))
    jobs = []
    # Add tuple jobs (shared index, unique subid)
    for idx, (ts_str, files) in enumerate(sorted(tuple_groups.items())):
        # Sort files by type for deterministic subid assignment
        type_order = ['left', 'right', 'out']
        files_sorted = sorted(files, key=lambda x: type_order.index(x[2]) if x[2] in type_order else 99)
        for subidx, (orig_path, base_name, file_type, ts_str, ext) in enumerate(files_sorted):
            subid = TUPLE_SUBID.get(file_type, chr(100 + subidx))  # fallback: d, e, ...
            job_data = {
                'orig_path': str(orig_path),
                'base_name': base_name,
                'is_tuple': True,
                'tuple_index': idx,
                'subid': subid,
                'file_type': file_type,
                'timestamp': ts_str,
                'ext': ext
            }
            jobs.append(Job(job_id=f"tuple_{idx}_{subid}_{file_type}", data=job_data))
    # Add single jobs
    for idx, (orig_path, base_name, ts_str, ext) in enumerate(singles):
        job_data = {
            'orig_path': str(orig_path),
            'base_name': base_name,
            'is_tuple': False,
            'tuple_index': idx,
            'subid': None,
            'file_type': 'single',
            'timestamp': ts_str,
            'ext': ext
        }
        jobs.append(Job(job_id=f"single_{idx}", data=job_data))
    return jobs

def extract_type(filename):
    for key, mapped in TYPE_MAP.items():
        if key in filename:
            return mapped, key
    return None, None

# Patch process_file_job to use new filename format for tuples
def process_file_job_with_subid(job, run_folder: Path):
    from file_ingestion import process_file_job as orig_process_file_job
    # Patch the filename logic for tuples
    if job.data['is_tuple']:
        index_str = f"{job.data['tuple_index']:04d}"
        subid = job.data['subid']
        ts_str = job.data['timestamp']
        file_type = job.data['file_type']
        ext = job.data['ext']
        new_name = f"{index_str}-{subid}-{file_type}-{ts_str}{ext}"
        job.data['output_name'] = new_name
    return orig_process_file_job(job, run_folder)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Audio Context Tool Orchestrator")
    parser.add_argument('input_dir', type=str, help='Input directory')
    parser.add_argument('--asr_engine', type=str, choices=['parakeet', 'whisper'], default='parakeet', help='ASR engine to use for transcription (default: parakeet)')
    parser.add_argument('--llm_config', type=str, default=None, help='Path to LLM task config JSON (default: workflows/llm_tasks.json)')
    parser.add_argument('--call-tones', action='store_true', help='Append tones.wav to end of each call and between calls in show file')
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    run_ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_folder = Path('output') / run_ts
    run_folder.mkdir(parents=True, exist_ok=True)
    orchestrator = PipelineOrchestrator(run_folder)
    jobs = create_jobs_from_input(input_dir)
    # Use patched process_file_job for new filename format
    global process_file_job
    process_file_job = process_file_job_with_subid
    for job in jobs:
        orchestrator.add_job(job)
    # Run full pipeline (transcription and renaming included)
    orchestrator.run(call_tones=args.call_tones)
    # Run LLM stage after all other stages
    orchestrator.run_llm_stage(llm_config_path=args.llm_config)
    # No redundant transcription after renaming 