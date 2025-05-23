import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Callable, Dict, Any
from tqdm import tqdm
from rich.console import Console
from rich.traceback import install as rich_traceback_install
import uuid
import numpy as np
from file_ingestion import process_file_job, SUPPORTED_EXTENSIONS, TYPE_MAP
from audio_separation import separate_audio_file
from clap_annotation import annotate_clap_for_out_files, segment_audio_with_clap
from speaker_diarization import batch_diarize, segment_speakers_from_diarization
from transcription import transcribe_segments
import torchaudio
import re
import soundfile as sf
from collections import defaultdict
# --- Finalization stage import ---
from finalization_stage import run_finalization_stage
# --- Resume functionality imports ---
from resume_utils import add_resume_to_orchestrator, run_stage_with_resume, should_resume_pipeline, print_resume_status, print_stage_status
import random
import hashlib
import yaml
import tempfile
import shutil

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
    def __init__(self, run_folder: Path, asr_engine: str, llm_config_path: str = None):
        self.jobs: List[Job] = []
        self.log: List[Dict[str, Any]] = []
        self.manifest: List[Dict[str, Any]] = []
        self.run_folder = run_folder
        self._log_buffer: List[Dict[str, Any]] = []  # Buffer for log events
        self._console_buffer: List[str] = []         # Buffer for console output
        self._logging_enabled = False                # Only enable after PII is gone
        self.asr_engine = asr_engine
        self.llm_config_path = llm_config_path
        self.global_llm_seed = None
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
        # Ensure run_folder exists before writing
        self.run_folder.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_path = self.run_folder / f'orchestrator-log-{ts}.json'
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.log, f, indent=2)
        if self._logging_enabled:
            console.print(f"[bold green]Orchestrator log written to {log_path}[/]")
        else:
            self._console_buffer.append(f"[bold green]Orchestrator log written to {log_path}[/]")
    def write_manifest(self):
        # Ensure run_folder exists before writing  
        self.run_folder.mkdir(parents=True, exist_ok=True)
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
        Single files (like 'out' files) skip separation and are copied directly for diarization.
        """
        renamed_dir = self.run_folder / 'renamed'
        
        # Check if renamed directory exists
        if not renamed_dir.exists():
            self.log_event('WARNING', 'renamed_directory_missing', {
                'renamed_dir': str(renamed_dir),
                'message': 'No renamed directory found, skipping separation job creation'
            })
            return
        
        # Check if directory is empty
        renamed_files = list(renamed_dir.glob('*'))
        if not renamed_files:
            self.log_event('WARNING', 'renamed_directory_empty', {
                'renamed_dir': str(renamed_dir),
                'message': 'Renamed directory is empty, no separation jobs to create'
            })
            return
        
        separation_job_count = 0
        single_file_count = 0
        
        # Handle traditional left/right files that need separation
        for file in renamed_dir.iterdir():
            if file.is_file() and ('-left-' in file.name or '-right-' in file.name):
                job_data = {
                    'input_path': str(file),
                    'input_name': file.name
                }
                job_id = f"separate_{file.stem}"
                self.jobs.append(Job(job_id=job_id, data=job_data, job_type='separate'))
                separation_job_count += 1
        
        # Handle single files (complete conversations) - skip separation, prepare for diarization
        for file in renamed_dir.iterdir():
            if file.is_file() and not ('-left-' in file.name or '-right-' in file.name):
                # This is a complete conversation file - skip separation
                single_file_count += 1
                
                # Extract call_id from filename (the index part)
                call_id = file.stem.split('-')[0]  # e.g., "0000" from "0000-out-20241227-001220"
                
                # Create separated directory structure (even though we're not separating)
                separated_dir = self.run_folder / 'separated' / call_id
                separated_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy as complete conversation file for diarization
                conversation_file = separated_dir / f"{call_id}-conversation.wav"
                
                # Convert to WAV if needed and copy
                import soundfile as sf
                try:
                    # Read and convert to WAV
                    audio, sr = sf.read(str(file))
                    sf.write(str(conversation_file), audio, sr)
                    
                    self.log_event('INFO', 'single_file_ready_for_diarization', {
                        'input_file': str(file),
                        'output_file': str(conversation_file),
                        'call_id': call_id,
                        'message': 'Complete conversation file, skipped separation'
                    })
                except Exception as e:
                    self.log_event('ERROR', 'single_file_preparation_failed', {
                        'input_file': str(file),
                        'error': str(e)
                    })
        
        self.log_event('INFO', 'separation_jobs_created', {
            'separation_job_count': separation_job_count,
            'single_file_count': single_file_count,
            'total_files': len(renamed_files)
        })

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

    def run_clap_annotation_stage(self, annotation_config_path=None):
        """
        Run CLAP annotation for all 'out' files in renamed/ using config from workflows/clap_annotation.json.
        Updates manifest and logs via orchestrator methods.
        """
        renamed_dir = self.run_folder / 'renamed'
        clap_dir = self.run_folder / 'clap'
        import json
        config_path = annotation_config_path or Path('workflows/clap_annotation.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            annotation_config = json.load(f)
        prompts = annotation_config.get('prompts', [])
        confidence_threshold = annotation_config.get('confidence_threshold', 0.6)
        chunk_length_sec = annotation_config.get('chunk_length_sec', 5)
        overlap_sec = annotation_config.get('overlap_sec', 2)
        self.log_event('INFO', 'clap_annotation_start', {
            'prompts': prompts,
            'confidence_threshold': confidence_threshold,
            'chunk_length_sec': chunk_length_sec,
            'overlap_sec': overlap_sec,
            'config_path': str(config_path)
        })
        fused_clap_model = 'laion/clap-htsat-fused'
        results = annotate_clap_for_out_files(
            renamed_dir, clap_dir, prompts,
            model_id=fused_clap_model,
            chunk_length_sec=chunk_length_sec,
            overlap_sec=overlap_sec,
            confidence_threshold=confidence_threshold
        )
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
        Run speaker diarization for all vocal files in separated/<call id>/.
        Handles both traditional *-vocals.wav files and complete *-conversation.wav files.
        Sets max_speakers=8 for all files (model limit).
        Updates manifest and logs via orchestrator methods.
        Logs every file written and updates manifest.
        """
        separated_dir = self.run_folder / 'separated'
        diarized_dir = self.run_folder / 'diarized'
        diarized_dir.mkdir(exist_ok=True)
        self.log_event('INFO', 'diarization_start', {'max_speakers': 8})
        
        # Find all files to diarize (vocals from separation + complete conversations)
        audio_files = []
        for call_dir in separated_dir.iterdir():
            if not call_dir.is_dir():
                continue
            # Traditional pattern: left-vocals.wav, right-vocals.wav
            for vocals_file in call_dir.glob('*-vocals.wav'):
                audio_files.append(vocals_file)
            # Complete conversation pattern: *-conversation.wav
            for conv_file in call_dir.glob('*-conversation.wav'):
                audio_files.append(conv_file)
        
        if not audio_files:
            self.log_event('WARNING', 'no_audio_files_found', {
                'separated_dir': str(separated_dir),
                'message': 'No audio files found for diarization'
            })
        
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
        Skips and logs empty or unreadable files.
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
                try:
                    waveform, sr = torchaudio.load(str(wav_path))
                    print(f"[DEBUG] Loaded: {wav_path} shape={waveform.shape} sr={sr} dtype={waveform.dtype}")
                    if waveform.numel() == 0 or (waveform.ndim == 2 and waveform.shape[1] == 0):
                        warn_msg = f"Segment file is empty, skipping: {wav_path.name}"
                        print(f"[WARN] {warn_msg}")
                        self.log_and_manifest(
                            stage='resampled_skipped',
                            call_id=entry.get('call_id'),
                            input_files=[str(wav_path)],
                            output_files=[],
                            params={'reason': 'empty file'},
                            metadata=None,
                            event='file_skipped',
                            result='skipped',
                            error=warn_msg
                        )
                        continue
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
                except Exception as e:
                    warn_msg = f"Resample failed for {wav_path.name}: {str(e)}"
                    print(f"[WARN] {warn_msg}")
                    self.log_and_manifest(
                        stage='resampled_skipped',
                        call_id=entry.get('call_id'),
                        input_files=[str(wav_path)],
                        output_files=[],
                        params={'reason': 'exception'},
                        metadata=None,
                        event='file_skipped',
                        result='skipped',
                        error=warn_msg
                    )
                    continue
        self.log_event('INFO', 'resample_segments_complete', {'count': len(resampled_segments)})

    def run_transcription_stage(self, asr_engine='parakeet', asr_config=None):
        """
        Transcribe all speaker segments using the selected ASR engine (parakeet or whisper).
        Updates manifest with transcript results.
        Logs every file written and updates manifest.
        Tracks and logs failed/skipped transcriptions for full auditability.
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
        success_count = 0
        fail_count = 0
        skip_count = 0
        for seg, res in zip(segments, results):
            if res.get('error'):
                self.log_and_manifest(
                    stage='transcription_failed',
                    call_id=seg.get('call_id'),
                    input_files=[str(seg.get('wav'))],
                    output_files=[],
                    params={'asr_engine': asr_engine},
                    metadata={'error': res.get('error')},
                    event='transcription',
                    result='error',
                    error=res.get('error')
                )
                fail_count += 1
                continue
            # Check for missing .txt or .json
            if not res.get('txt') or not os.path.exists(res.get('txt')):
                warn_msg = f"Transcription .txt missing for {seg.get('wav')}"
                self.log_and_manifest(
                    stage='transcription_skipped',
                    call_id=seg.get('call_id'),
                    input_files=[str(seg.get('wav'))],
                    output_files=[],
                    params={'asr_engine': asr_engine},
                    metadata={'warning': warn_msg},
                    event='transcription',
                    result='skipped',
                    error=warn_msg
                )
                skip_count += 1
                continue
            self.log_and_manifest(
                stage='transcribed',
                call_id=res.get('call_id'),
                input_files=[str(res.get('wav'))],
                output_files=[str(res.get('txt')), str(res.get('json'))] if res.get('txt') and res.get('json') else [],
                params={'asr_engine': asr_engine},
                metadata={'error': res.get('error') if 'error' in res else None},
                event='transcription',
                result='success',
                error=res.get('error') if 'error' in res else None
            )
            self.manifest.append({
                **res,
                'stage': 'transcribed'
            })
            success_count += 1
        self.log_event('INFO', 'transcription_complete', {'count': len(results), 'success': success_count, 'failed': fail_count, 'skipped': skip_count})
        print(f"[SUMMARY] Transcription: {success_count} succeeded, {fail_count} failed, {skip_count} skipped.")

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
                # Only include segments for this call_id (exclude [0000-CONVERSATION] etc.)
                if s.get('call_id') != call_id:
                    continue
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

    def run_llm_task_for_call(self, call_id, master_transcript, llm_config, output_dir, llm_tasks, global_llm_seed=None):
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
            # Determine seed
            if global_llm_seed is not None:
                # Deterministic per-task seed from global seed, call_id, and task name
                seed_input = f"{global_llm_seed}_{call_id}_{name}"
                seed = int(hashlib.sha256(seed_input.encode()).hexdigest(), 16) % (2**32)
            else:
                seed = random.randint(0, 2**32-1)
            data = {
                "model": model_id,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed  # Pass seed if LLM API supports it
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
                    self.log_event('ERROR', 'llm_api_error', {'call_id': call_id, 'task': name, 'status': response.status_code, 'text': response.text, 'seed': seed})
            except Exception as e:
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(f"LLM request failed: {e}")
                output_paths[name] = str(out_path)
                self.log_event('ERROR', 'llm_request_failed', {'call_id': call_id, 'task': name, 'error': str(e), 'seed': seed})
            # Always log the seed used
            self.log_event('INFO', 'llm_task_seed', {'call_id': call_id, 'task': name, 'seed': seed, 'output_file': str(out_path)})
        return output_paths

    def run_llm_stage(self, llm_config_path=None):
        import json
        from pathlib import Path
        # --- Read pipeline mode from pipeline_state.json if available ---
        mode = 'call'  # default
        pipeline_state_path = self.run_folder / 'pipeline_state.json'
        if pipeline_state_path.exists():
            try:
                with open(pipeline_state_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                if 'mode' in state:
                    mode = state['mode']
            except Exception as e:
                self.log_event('WARNING', 'pipeline_state_read_failed', {'error': str(e)})
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
        llm_dir.mkdir(exist_ok=True, parents=True)
        call_ids = [d.name for d in soundbites_dir.iterdir() if d.is_dir()]
        self.log_event('INFO', 'llm_stage_start', {'llm_dir': str(llm_dir), 'mode': mode})
        max_tokens = 16384
        safe_chunk = 10000  # chunk size for LLM input
        for call_id in call_ids:
            master_transcript = self.get_master_transcript_path(call_id)
            transcript_text = None
            if master_transcript.exists():
                with open(master_transcript, 'r', encoding='utf-8') as f:
                    transcript_text = f.read()
            call_llm_dir = llm_dir / call_id
            call_llm_dir.mkdir(parents=True, exist_ok=True)
            if mode == 'call':
                # Only run LLM tasks on the master transcript for each call
                for task in llm_tasks:
                    name = task.get('name', 'unnamed_task')
                    prompt_template = task.get('prompt_template', '')
                    output_file = f'{name}.txt'
                    output_path = call_llm_dir / output_file
                    prompt = prompt_template.format(transcript=transcript_text)
                    # LLM API call (same as run_llm_task_for_call)
                    import requests, random, hashlib
                    base_url = llm_config.get('lm_studio_base_url', 'http://localhost:1234/v1')
                    api_key = llm_config.get('lm_studio_api_key', 'lm-studio')
                    model_id = llm_config.get('lm_studio_model_identifier', 'llama-3.1-8b-supernova-etherealhermes')
                    temperature = llm_config.get('lm_studio_temperature', 0.5)
                    max_tokens = llm_config.get('lm_studio_max_tokens', 250)
                    headers = {
                        'Authorization': f'Bearer {api_key}',
                        'Content-Type': 'application/json'
                    }
                    seed_input = f"{call_id}_{name}"
                    seed = int(hashlib.sha256(seed_input.encode()).hexdigest(), 16) % (2**32)
                    data = {
                        "model": model_id,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "seed": seed
                    }
                    try:
                        response = requests.post(f"{base_url}/chat/completions", headers=headers, data=json.dumps(data), timeout=60)
                        if response.status_code == 200:
                            result = response.json()
                            content = result['choices'][0]['message']['content'].strip()
                            with open(output_path, 'w', encoding='utf-8') as f:
                                f.write(content)
                        else:
                            error_msg = f"LLM API error {response.status_code}: {response.text}"
                            with open(output_path, 'w', encoding='utf-8') as f:
                                f.write(error_msg)
                            self.log_event('ERROR', 'llm_api_error', {'call_id': call_id, 'task': name, 'status': response.status_code, 'text': response.text, 'seed': seed})
                    except Exception as e:
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(f"LLM request failed: {e}")
                        self.log_event('ERROR', 'llm_request_failed', {'call_id': call_id, 'task': name, 'error': str(e), 'seed': seed})
                    self.log_and_manifest(
                        stage='llm',
                        call_id=call_id,
                        input_files=[str(master_transcript)],
                        output_files=[str(output_path)],
                        params={'task': name, 'prompt_template': prompt_template},
                        metadata={'llm_model': llm_config.get('lm_studio_model_identifier')},
                        event='llm_task',
                        result='success'
                    )
                # Warn if per-speaker LLM tasks are configured in call mode
                per_speaker = self.get_per_speaker_transcripts(call_id)
                if per_speaker:
                    self.log_event('WARNING', 'per_speaker_llm_skipped', {'call_id': call_id, 'mode': mode, 'message': 'Per-speaker LLM tasks are skipped in call mode.'})
            else:
                # Single-file mode: run per-speaker LLM tasks as before
                per_speaker = self.get_per_speaker_transcripts(call_id)
                speaker_outputs = {}
                for speaker_id, speaker_text in per_speaker.items():
                    speaker_outputs[speaker_id] = {}
                    token_count = estimate_tokens(speaker_text)
                    if token_count > safe_chunk:
                        # Chunking logic (as before)
                        words = speaker_text.split()
                        chunk_size = safe_chunk * 4
                        chunks = []
                        current = []
                        current_len = 0
                        for word in words:
                            current.append(word)
                            current_len += len(word) + 1
                            if current_len >= chunk_size:
                                chunks.append(' '.join(current))
                                current = []
                                current_len = 0
                        if current:
                            chunks.append(' '.join(current))
                        for task in llm_tasks:
                            name = task.get('name', 'unnamed_task')
                            prompt_template = task.get('prompt_template', '')
                            output_file = f'{speaker_id}_{name}.txt'
                            output_path = call_llm_dir / output_file
                            responses = []
                            for i, chunk in enumerate(chunks):
                                prompt = f"[Part {i+1} of {len(chunks)} for Speaker {speaker_id}]\n" + prompt_template.format(transcript=chunk)
                                import requests, random, hashlib
                                base_url = llm_config.get('lm_studio_base_url', 'http://localhost:1234/v1')
                                api_key = llm_config.get('lm_studio_api_key', 'lm-studio')
                                model_id = llm_config.get('lm_studio_model_identifier', 'llama-3.1-8b-supernova-etherealhermes')
                                temperature = llm_config.get('lm_studio_temperature', 0.5)
                                max_tokens = llm_config.get('lm_studio_max_tokens', 250)
                                headers = {
                                    'Authorization': f'Bearer {api_key}',
                                    'Content-Type': 'application/json'
                                }
                                seed_input = f"{call_id}_{speaker_id}_{name}_{i}"
                                seed = int(hashlib.sha256(seed_input.encode()).hexdigest(), 16) % (2**32)
                                data = {
                                    "model": model_id,
                                    "messages": [
                                        {"role": "user", "content": prompt}
                                    ],
                                    "temperature": temperature,
                                    "max_tokens": max_tokens,
                                    "seed": seed
                                }
                                try:
                                    response = requests.post(f"{base_url}/chat/completions", headers=headers, data=json.dumps(data), timeout=60)
                                    if response.status_code == 200:
                                        result = response.json()
                                        content = result['choices'][0]['message']['content'].strip()
                                        responses.append(content)
                                    else:
                                        error_msg = f"LLM API error {response.status_code}: {response.text}"
                                        responses.append(error_msg)
                                        self.log_event('ERROR', 'llm_api_error', {'call_id': call_id, 'speaker_id': speaker_id, 'task': name, 'status': response.status_code, 'text': response.text, 'seed': seed})
                                except Exception as e:
                                    responses.append(f"LLM request failed: {e}")
                                    self.log_event('ERROR', 'llm_request_failed', {'call_id': call_id, 'speaker_id': speaker_id, 'task': name, 'error': str(e), 'seed': seed})
                            final_response = '\n\n'.join(responses)
                            with open(output_path, 'w', encoding='utf-8') as f:
                                f.write(final_response)
                            speaker_outputs[speaker_id][name] = str(output_path)
                            self.log_and_manifest(
                                stage='llm',
                                call_id=call_id,
                                input_files=[f'per-speaker:{speaker_id}'],
                                output_files=[str(output_path)],
                                params={'task': name, 'prompt_template': prompt_template, 'speaker_id': speaker_id, 'chunks': len(chunks)},
                                metadata={'llm_model': llm_config.get('lm_studio_model_identifier')},
                                event='llm_task',
                                result='success'
                            )
                    else:
                        for task in llm_tasks:
                            name = task.get('name', 'unnamed_task')
                            prompt_template = task.get('prompt_template', '')
                            output_file = f'{speaker_id}_{name}.txt'
                            output_path = call_llm_dir / output_file
                            prompt = prompt_template.format(transcript=speaker_text)
                            import requests, random, hashlib
                            base_url = llm_config.get('lm_studio_base_url', 'http://localhost:1234/v1')
                            api_key = llm_config.get('lm_studio_api_key', 'lm-studio')
                            model_id = llm_config.get('lm_studio_model_identifier', 'llama-3.1-8b-supernova-etherealhermes')
                            temperature = llm_config.get('lm_studio_temperature', 0.5)
                            max_tokens = llm_config.get('lm_studio_max_tokens', 250)
                            headers = {
                                'Authorization': f'Bearer {api_key}',
                                'Content-Type': 'application/json'
                            }
                            seed_input = f"{call_id}_{speaker_id}_{name}"
                            seed = int(hashlib.sha256(seed_input.encode()).hexdigest(), 16) % (2**32)
                            data = {
                                "model": model_id,
                                "messages": [
                                    {"role": "user", "content": prompt}
                                ],
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                                "seed": seed
                            }
                            try:
                                response = requests.post(f"{base_url}/chat/completions", headers=headers, data=json.dumps(data), timeout=60)
                                if response.status_code == 200:
                                    result = response.json()
                                    content = result['choices'][0]['message']['content'].strip()
                                    with open(output_path, 'w', encoding='utf-8') as f:
                                        f.write(content)
                                else:
                                    error_msg = f"LLM API error {response.status_code}: {response.text}"
                                    with open(output_path, 'w', encoding='utf-8') as f:
                                        f.write(error_msg)
                                    self.log_event('ERROR', 'llm_api_error', {'call_id': call_id, 'speaker_id': speaker_id, 'task': name, 'status': response.status_code, 'text': response.text, 'seed': seed})
                            except Exception as e:
                                with open(output_path, 'w', encoding='utf-8') as f:
                                    f.write(f"LLM request failed: {e}")
                                self.log_event('ERROR', 'llm_request_failed', {'call_id': call_id, 'speaker_id': speaker_id, 'task': name, 'error': str(e), 'seed': seed})
                            speaker_outputs[speaker_id][name] = str(output_path)
                            self.log_and_manifest(
                                stage='llm',
                                call_id=call_id,
                                input_files=[f'per-speaker:{speaker_id}'],
                                output_files=[str(output_path)],
                                params={'task': name, 'prompt_template': prompt_template, 'speaker_id': speaker_id},
                                metadata={'llm_model': llm_config.get('lm_studio_model_identifier')},
                                event='llm_task',
                                result='success'
                            )
                # Aggregate per-speaker outputs into a summary file
                agg_path = call_llm_dir / 'per_speaker_llm_outputs.json'
                with open(agg_path, 'w', encoding='utf-8') as f:
                    json.dump(speaker_outputs, f, indent=2)
                self.log_and_manifest(
                    stage='llm',
                    call_id=call_id,
                    input_files=None,
                    output_files=[str(agg_path)],
                    params={'aggregation': 'per-speaker'},
                    metadata={'speaker_outputs': speaker_outputs},
                    event='llm_aggregation',
                    result='success'
                )

    def run_normalization_stage(self):
        """
        Normalize separated vocal stems to -14.0 LUFS and output to normalized/<call id>/<channel>.wav.
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

    def run_true_peak_normalization_stage(self):
        """
        Apply true peak normalization to prevent digital clipping on normalized vocals.
        Uses -1.0 dBTP (decibels True Peak) limit which is broadcast standard.
        Logs every file written and updates manifest.
        """
        import pyloudnorm as pyln
        import soundfile as sf
        from pathlib import Path
        
        normalized_dir = self.run_folder / 'normalized'
        true_peak_dir = self.run_folder / 'true_peak_normalized'
        true_peak_dir.mkdir(exist_ok=True)
        
        self.log_event('INFO', 'true_peak_normalization_start', {'target_dbtp': -1.0})
        
        for call_id in os.listdir(normalized_dir):
            call_norm_dir = normalized_dir / call_id
            if not call_norm_dir.is_dir():
                continue
            
            call_tp_dir = true_peak_dir / call_id
            call_tp_dir.mkdir(parents=True, exist_ok=True)
            
            # Process all vocal files (both traditional and conversation)
            for vocal_file in call_norm_dir.glob('*.wav'):
                src = vocal_file
                dst = call_tp_dir / vocal_file.name
                
                try:
                    audio, sr = sf.read(str(src))
                    
                    # Ensure mono for processing
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    
                    # Apply true peak limiting to -1.0 dBTP
                    meter = pyln.Meter(sr)
                    peak_normalized = pyln.normalize.peak(audio, -1.0)
                    
                    # Save the true peak normalized audio
                    sf.write(str(dst), peak_normalized, sr)
                    
                    # Calculate true peak level for logging
                    true_peak_db = 20 * np.log10(np.max(np.abs(peak_normalized)))
                    
                    self.log_and_manifest(
                        stage='true_peak_normalized',
                        call_id=call_id,
                        input_files=[str(src)],
                        output_files=[str(dst)],
                        params={'target_dbtp': -1.0},
                        metadata={'measured_true_peak_db': true_peak_db},
                        event='file_written',
                        result='success'
                    )
                    
                except Exception as e:
                    self.log_event('ERROR', 'true_peak_normalization_failed', {
                        'call_id': call_id,
                        'file': vocal_file.name,
                        'error': str(e)
                    })
        
        self.log_event('INFO', 'true_peak_normalization_complete', {'true_peak_dir': str(true_peak_dir)})

    def run_remix_stage(self, call_tones=False):
        """
        For each call, mix true peak normalized vocals + 50% instrumental for each channel, then combine with 60/40 stereo panning:
        stereo_left = 0.6 * left_mix + 0.4 * right_mix
        stereo_right = 0.4 * left_mix + 0.6 * right_mix
        Output stereo remixed_call.wav. Tones are NOT appended here; tones are handled in the show stage only.
        Logs every file written and updates manifest.
        Always uses true peak normalized vocals for best audio quality.
        """
        import soundfile as sf
        import numpy as np
        from pathlib import Path
        
        # Use true peak normalized vocals (if available) or fall back to regular normalized
        true_peak_dir = self.run_folder / 'true_peak_normalized'
        normalized_dir = self.run_folder / 'normalized'
        vocals_dir = true_peak_dir if true_peak_dir.exists() else normalized_dir
        
        separated_dir = self.run_folder / 'separated'
        call_dir = self.run_folder / 'call'
        call_dir.mkdir(exist_ok=True)
        tones_path = Path('tones.wav')
        
        self.log_event('INFO', 'remix_start', {
            'vocals_source': 'true_peak_normalized' if vocals_dir == true_peak_dir else 'normalized',
            'vocals_dir': str(vocals_dir)
        })
        
        for call_id in os.listdir(vocals_dir):
            call_vocals_dir = vocals_dir / call_id
            call_sep_dir = separated_dir / call_id
            if not call_vocals_dir.is_dir() or not call_sep_dir.is_dir():
                continue
            out_call_dir = call_dir / call_id
            out_call_dir.mkdir(parents=True, exist_ok=True)
            
            # Handle both traditional stereo calls and single conversation files
            if (call_vocals_dir / "left-vocals.wav").exists() and (call_vocals_dir / "right-vocals.wav").exists():
                # Traditional stereo call processing
                left_vocal_path = call_vocals_dir / "left-vocals.wav"
                right_vocal_path = call_vocals_dir / "right-vocals.wav"
                left_instr_path = call_sep_dir / "left-instrumental.wav"
                right_instr_path = call_sep_dir / "right-instrumental.wav"
                
                if not (left_instr_path.exists() and right_instr_path.exists()):
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
                
                input_files = [str(left_vocal_path), str(right_vocal_path), str(left_instr_path), str(right_instr_path)]
                
            else:
                # Single conversation file processing (mono to stereo)
                conversation_vocal = None
                for vocal_file in call_vocals_dir.glob('*.wav'):
                    conversation_vocal = vocal_file
                    break
                
                if not conversation_vocal:
                    continue
                
                # Read conversation audio and create stereo mix
                vocal_audio, sr = sf.read(str(conversation_vocal))
                if vocal_audio.ndim > 1:
                    vocal_audio = vocal_audio.mean(axis=1)
                
                # Create stereo from mono (no instrumental for single files)
                stereo = np.stack([vocal_audio, vocal_audio], axis=-1)
                input_files = [str(conversation_vocal)]
            
            remixed_path = out_call_dir / "remixed_call.wav"
            sf.write(str(remixed_path), stereo, sr)
            
            self.log_and_manifest(
                stage='remix',
                call_id=call_id,
                input_files=input_files,
                output_files=[str(remixed_path)],
                params={'panning': '60/40 split' if len(input_files) > 2 else 'mono to stereo'},
                metadata={'vocals_source': 'true_peak_normalized' if vocals_dir == true_peak_dir else 'normalized'},
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
        If call_tones is set, insert tones.wav between calls (not after the last call).
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
            # Insert tones between calls if call_tones is set and not the last call
            if call_tones and idx < len(call_files) - 1 and tones_path.exists():
                tones, tones_sr = sf.read(str(tones_path))
                if tones_sr != sr:
                    import librosa
                    tones = librosa.resample(tones.T, orig_sr=tones_sr, target_sr=sr).T
                tones_start = cur_time
                tones_end = tones_start + tones.shape[0] / sr
                show_audio.append(tones)
                show_timeline.append({
                    'tones': True,
                    'start': tones_start,
                    'end': tones_end
                })
                cur_time = tones_end
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

    def run_finalization_stage(self):
        """
        Run the finalization stage - converts to MP3, embeds ID3 tags, handles LLM output
        """
        run_finalization_stage(self.run_folder, self.manifest)

    def run(self, mode='auto', call_cutter=False):
        print(f"[INFO] Running pipeline in {mode} mode.")
        if mode == 'single-file':
            # Run single-file workflow (CLAP segmentation, annotation, etc.)
            self.run_single_file_workflow()
        elif mode == 'calls':
            # Run call-processing workflow
            self.run_call_processing_workflow()
        else:
            # Fallback/auto
            self.run_call_processing_workflow()

    def run_single_file_workflow(self):
        # Full pipeline for single-file mode
        self._run_ingestion_jobs()
        # Find the original input audio file (should be in renamed/)
        renamed_dir = self.run_folder / 'renamed'
        input_files = [f for f in renamed_dir.iterdir() if f.is_file()]
        if not input_files:
            print("[ERROR] No input files found in renamed/ directory.")
            return
        original_audio = input_files[0]
        print(f"[DEBUG] Using original audio for CLAP: {original_audio}")
        # Run CLAP segmentation on original audio
        segmented_dir = self.run_folder / 'segmented'
        segmented_dir.mkdir(parents=True, exist_ok=True)
        from clap_annotation import segment_audio_with_clap
        import json
        segmentation_config_path = Path('workflows/clap_segmentation.json')
        with open(segmentation_config_path, 'r', encoding='utf-8') as f:
            segmentation_config = json.load(f)["clap_segmentation"]
        segments = segment_audio_with_clap(
            original_audio,
            segmentation_config,
            segmented_dir,
            model_id=segmentation_config.get("model_id", "laion/clap-htsat-unfused"),
            chunk_length_sec=segmentation_config.get("chunk_length_sec", 5),
            overlap_sec=segmentation_config.get("overlap_sec", 2)
        )
        # If segments found, process each segment; else, process the whole file
        files_to_process = []
        if segments:
            print(f"[DEBUG] CLAP segmentation found {len(segments)} segments.")
            for seg in segments:
                files_to_process.append(Path(seg["output_path"]))
        else:
            print("[DEBUG] No CLAP segments found; processing whole file.")
            files_to_process = [original_audio]
        # For each file (segment or whole), perform separation and downstream steps
        for idx, audio_file in enumerate(files_to_process):
            print(f"[DEBUG] Processing segment {idx}: {audio_file}")
            # Place segment in a per-segment input folder for separation
            seg_input_dir = self.run_folder / 'segment_inputs' / f'segment_{idx:04d}'
            seg_input_dir.mkdir(parents=True, exist_ok=True)
            seg_audio_path = seg_input_dir / audio_file.name
            import shutil
            shutil.copy2(audio_file, seg_audio_path)
            # Ingest segment as a job
            job = Job(job_id=f'segment_{idx:04d}', data={
                'orig_path': str(seg_audio_path),
                'base_name': seg_audio_path.name,
                'is_tuple': False,
                'tuple_index': idx,
                'subid': 'c',
                'file_type': 'out',
                'timestamp': '',
                'ext': seg_audio_path.suffix
            })
            self.jobs = [job]  # Overwrite jobs for this segment
            self.add_separation_jobs()
            self.run_audio_separation_stage()
            self.run_normalization_stage()
            self.run_true_peak_normalization_stage()
            # Use separated vocals for downstream steps
            separated_dir = self.run_folder / 'separated' / f'{idx:04d}'
            vocal_path = separated_dir / 'out-vocals.wav'
            if not vocal_path.exists():
                wavs = list(separated_dir.glob('*.wav'))
                if not wavs:
                    print(f"[ERROR] No separated vocals found for segment {idx}.")
                    continue
                vocal_path = wavs[0]
            # Run CLAP annotation on the separated vocal (for context, not segmentation)
            from clap_annotation import annotate_clap_for_out_files
            clap_dir = self.run_folder / 'clap' / f'{idx:04d}'
            annotate_clap_for_out_files(
                input_dir=Path(vocal_path).parent,
                output_dir=clap_dir,
                prompts=segmentation_config.get("prompts", []),
                model_id=segmentation_config.get("model_id", "laion/clap-htsat-unfused"),
                chunk_length_sec=segmentation_config.get("chunk_length_sec", 5),
                overlap_sec=segmentation_config.get("overlap_sec", 2),
                confidence_threshold=segmentation_config.get("confidence_threshold", 0.6)
            )
            # Downstream steps
            self.run_diarization_stage()
            self.run_speaker_segmentation_stage()
            self.run_resample_segments_stage()
            self.run_transcription_stage(asr_engine=self.asr_engine)
            self.run_rename_soundbites_stage()
            self.run_final_soundbite_stage()
            self.run_llm_stage(llm_config_path=self.llm_config_path)
            self.run_remix_stage()
            self.run_show_stage()
            self.run_finalization_stage()

    def run_call_processing_workflow(self):
        # Full pipeline for call-processing mode
        self._run_ingestion_jobs()
        self.add_separation_jobs()
        self.run_audio_separation_stage()
        self.run_normalization_stage()
        self.run_true_peak_normalization_stage()
        self.run_clap_annotation_stage()
        self.run_diarization_stage()
        self.run_speaker_segmentation_stage()
        self.run_resample_segments_stage()
        self.run_transcription_stage(asr_engine=self.asr_engine)
        self.run_rename_soundbites_stage()
        self.run_final_soundbite_stage()
        self.run_llm_stage(llm_config_path=self.llm_config_path)
        self.run_remix_stage()
        self.run_show_stage()
        self.run_finalization_stage()

    def run_with_resume(self, call_tones=False, resume=True, resume_from=None):
        # Add resume functionality to orchestrator
        add_resume_to_orchestrator(self, resume_mode=True)
        # Print resume summary at start
        print_resume_status(self.run_folder, detailed=True)
        # Define all stages with their methods
        stages_and_methods = [
            ('ingestion', self._run_ingestion_jobs),
            ('separation', lambda: (self.add_separation_jobs(), self.run_audio_separation_stage())),
            ('normalization', self.run_normalization_stage),
            ('true_peak_normalization', self.run_true_peak_normalization_stage),
            ('clap', self.run_clap_annotation_stage),
            ('diarization', self.run_diarization_stage),
            ('segmentation', self.run_speaker_segmentation_stage),
            ('resampling', self.run_resample_segments_stage),
            ('transcription', lambda: self.run_transcription_stage(asr_engine=self.asr_engine)),
            ('soundbite_renaming', self.run_rename_soundbites_stage),
            ('soundbite_finalization', self.run_final_soundbite_stage),
            ('llm', self.run_llm_stage),
            ('remix', lambda: self.run_remix_stage(call_tones=call_tones)),
            ('show', lambda: self.run_show_stage(call_tones=call_tones)),
            ('finalization', self.run_finalization_stage)
        ]
        # Run each stage with resume support
        for stage_name, stage_method in stages_and_methods:
            try:
                run_stage_with_resume(self, stage_name, stage_method, resume_from)
            except Exception as e:
                self.log_event('ERROR', f'pipeline_failed_at_stage', {
                    'stage': stage_name,
                    'error': str(e)
                })
                console.print(f"\n[bold red]Pipeline failed at stage: {stage_name}[/]")
                console.print(f"[red]Error: {e}[/]")
                console.print(f"\n[yellow]To resume from this point, run with --resume[/]")
                raise
        # Write final manifest and log
        self.write_manifest()
        self.write_log()
        # Final summary
        print("\n\033[92m🎉🎉 Pipeline completed successfully! All outputs are ready. 🎉🎉\033[0m\n")
        print_resume_status(self.run_folder, detailed=True)

    def _run_ingestion_jobs(self):
        """Helper method for ingestion stage that can be used with resume functionality"""
        
        # Debug: Log job information
        total_jobs = len(self.jobs)
        rename_jobs = [job for job in self.jobs if job.job_type == 'rename']
        
        self.log_event('INFO', 'ingestion_start', {
            'total_jobs': total_jobs,
            'rename_jobs': len(rename_jobs),
            'job_details': [{'job_id': j.job_id, 'job_type': j.job_type, 'orig_path': j.data.get('orig_path')} for j in self.jobs[:5]]  # First 5 jobs
        })
        
        if total_jobs == 0:
            self.log_event('WARNING', 'no_jobs_to_process', {'message': 'No ingestion jobs found'})
            return
        
        with tqdm(total=len(self.jobs), desc="Ingestion Progress", position=0) as global_bar:
            processed_files = 0
            for job in self.jobs:
                if job.job_type == 'rename':
                    result = process_file_job(job, self.run_folder)
                    
                    # Debug: Log each file processing result
                    self.log_event('INFO', 'file_processing_result', {
                        'job_id': job.job_id,
                        'success': result.get('success', False),
                        'output_name': result.get('output_name'),
                        'output_path': result.get('output_path'),
                        'error': result.get('error')
                    })
                    
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
                        processed_files += 1
                    else:
                        self.log_event('ERROR', 'file_rename_failed', {
                            'output_name': result.get('output_name'),
                            'output_path': result.get('output_path'),
                            'error': result.get('error')
                        })
                        self.manifest.append(manifest_entry)
                    global_bar.update(1)
        
        self.log_event('INFO', 'ingestion_complete', {
            'total_jobs': total_jobs,
            'processed_files': processed_files,
            'renamed_dir': str(self.run_folder / 'renamed')
        })
        
        self.enable_logging()
        self.write_log()
        self.write_manifest()

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

    def run_clap_segmentation_stage(self, segmentation_config_path=None):
        """
        Run CLAP-based segmentation for long audio files if --call-cutter is set.
        Loads config from workflows/clap_segmentation.json by default.
        """
        renamed_dir = self.run_folder / 'renamed'
        segmented_dir = self.run_folder / 'segmented'
        segmented_dir.mkdir(exist_ok=True)
        # Load config
        import json
        config_path = segmentation_config_path or Path('workflows/clap_segmentation.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            segmentation_config = json.load(f)["clap_segmentation"]
        # Find eligible files (e.g., out- files longer than min_segment_length)
        min_len = segmentation_config.get("min_segment_length_sec", 10)
        for file in renamed_dir.iterdir():
            if not file.is_file() or '-out-' not in file.name:
                continue
            import soundfile as sf
            try:
                info = sf.info(str(file))
                duration = info.duration
            except Exception:
                continue
            if duration < min_len:
                continue
            call_id, channel, timestamp = file.stem.split('-')[0], 'out', None
            out_dir = segmented_dir / call_id
            out_dir.mkdir(parents=True, exist_ok=True)
            segments = segment_audio_with_clap(file, segmentation_config, out_dir)
            for seg in segments:
                self.log_and_manifest(
                    stage='clap_segmented',
                    call_id=call_id,
                    input_files=[str(file)],
                    output_files=[seg["output_path"]],
                    params={'start': seg['start'], 'end': seg['end']},
                    metadata={'events': seg['events']},
                    event='clap_segment',
                    result='success'
                )
                self.manifest.append({
                    'stage': 'clap_segmented',
                    'call_id': call_id,
                    'input_name': file.name,
                    'segment_index': seg['segment_index'],
                    'start': seg['start'],
                    'end': seg['end'],
                    'output_path': seg['output_path'],
                    'events': seg['events']
                })
        self.log_event('INFO', 'clap_segmentation_complete', {'segmented_dir': str(segmented_dir)})

    def get_per_speaker_transcripts(self, call_id):
        """
        Extract per-speaker transcripts for a call from speakers/ and diarized/ folders.
        Returns a dict: {speaker_id: transcript_text}
        """
        speakers_dir = self.run_folder / 'speakers' / call_id
        transcripts = {}
        if not speakers_dir.exists():
            return transcripts
        for channel_dir in speakers_dir.iterdir():
            if not channel_dir.is_dir():
                continue
            for speaker_dir in channel_dir.iterdir():
                if not speaker_dir.is_dir():
                    continue
                speaker_id = speaker_dir.name
                utterances = []
                for txt_file in sorted(speaker_dir.glob('*.txt')):
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        if text:
                            utterances.append(text)
                if utterances:
                    transcripts[speaker_id] = '\n'.join(utterances)
        return transcripts

def create_jobs_from_input(input_path: Path) -> List[Job]:
    """Create jobs from input path (can be a single file or directory)
    IMPORTANT: No PII (original filenames or paths) may be printed or logged here! Only output anonymized counts if needed.
    """
    all_files = []
    seen_names = set()
    
    # Handle both single files and directories
    if input_path.is_file():
        # Single file input
        file = input_path.name
        ext = input_path.suffix.lower()
        if ext in SUPPORTED_EXTENSIONS:
            orig_path = input_path
            base_name = file
            all_files.append((orig_path, base_name))
    elif input_path.is_dir():
        # Directory input - original logic
        for root, _, files in os.walk(input_path):
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
    else:
        # Input path doesn't exist or is neither file nor directory
        # No PII in error message
        print(f"❌ Input path does not exist or is invalid.")
        return []
    
    if not all_files:
        print(f"❌ No supported audio files found in input.")
        print(f"Supported extensions: {SUPPORTED_EXTENSIONS}")
        return []
    
    # Only output anonymized count, never filenames or paths
    print(f"🔧 Created {len(all_files)} job(s) for processing.")
    
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
            # Treat single audio files as 'out' files for full pipeline processing
            file_type = 'out'  # Force single files to be processed as 'out' files
            tuple_key = ts_str
            if tuple_key not in tuple_groups:
                tuple_groups[tuple_key] = []
            tuple_groups[tuple_key].append((orig_path, base_name, file_type, ts_str, ext))
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
    # Note: No separate single jobs now - all files are processed as tuples
    print(f"🔧 Created {len(jobs)} anonymized job(s): {len(tuple_groups)} tuple groups.")
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

# Utility to estimate token count
try:
    import tiktoken
    def estimate_tokens(text, model="gpt-3.5-turbo"):
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
except ImportError:
    def estimate_tokens(text, model=None):
        # Fallback: 1 token ≈ 4 chars (OpenAI heuristic)
        return max(1, len(text) // 4)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Audio Context Tool Orchestrator")
    parser.add_argument('input_dir', type=str, nargs='?', help='Input directory (required for fresh runs, optional when resuming)')
    parser.add_argument('--output-folder', type=str, help='Existing output folder to resume from (e.g., outputs/run-20250522-211044)')
    parser.add_argument('--asr_engine', type=str, choices=['parakeet', 'whisper'], default='parakeet', help='ASR engine to use for transcription (default: parakeet)')
    parser.add_argument('--llm_config', type=str, default=None, help='Path to LLM task config JSON (default: workflows/llm_tasks.json)')
    parser.add_argument('--llm-seed', type=int, default=None, help='Global seed for LLM tasks (default: random per task)')
    parser.add_argument('--call-tones', action='store_true', help='Append tones.wav to end of each call and between calls in show file')
    parser.add_argument('--call-cutter', action='store_true', help='Enable CLAP-based call segmentation for long files')
    parser.add_argument('--url', action='append', help='Download and process audio from one or more URLs')
    parser.add_argument('--mode', type=str, choices=['auto', 'single-file', 'calls'], default='auto', help='Workflow mode: auto (default), single-file, or calls')
    # Resume functionality arguments
    parser.add_argument('--resume', action='store_true', help='Enable resume functionality - skip completed stages')
    parser.add_argument('--resume-from', type=str, metavar='STAGE', help='Resume from a specific stage (skip all prior completed stages)')
    parser.add_argument('--force-rerun', type=str, metavar='STAGE', help='Force re-run a specific stage even if marked complete')
    parser.add_argument('--clear-from', type=str, metavar='STAGE', help='Clear completion status from specified stage onwards')
    parser.add_argument('--stage-status', type=str, metavar='STAGE', help='Show detailed status for a specific stage')
    parser.add_argument('--show-resume-status', action='store_true', help='Show current resume status and exit')
    parser.add_argument('--force', action='store_true', help='When used with --resume-from, deletes all outputs and state from that stage forward for a clean re-run')
    # 1. Add CLI flag for out-file processing
    parser.add_argument('--process-out-files', action='store_true', help='Enable processing of out- files as single-file inputs (lower fidelity, not recommended for main workflow)')

    # Print valid stage names in help
    from pipeline_state import get_pipeline_stages
    STAGE_LIST = get_pipeline_stages()
    parser.epilog = (
        "\nValid stage names for --resume-from, --force-rerun, --clear-from, --stage-status:\n  " + ", ".join(STAGE_LIST) +
        "\n\nExample: --resume-from diarization\n" +
        "\nUse --force with --resume-from to delete all outputs and state from that stage forward.\n" +
        "\nUse --llm-seed to set a global seed for LLM tasks (default: random per task).\n" +
        "\nUse --call-cutter to enable CLAP-based call segmentation for long files.\n" +
        "\n\nNote: By default, out- files are only used for CLAP segmentation/annotation. Use --process-out-files to process them as single-file inputs (not recommended for main workflow)."
    )

    args = parser.parse_args()

    # --- URL download logic ---
    url_files = []
    url_metadata = []
    if args.url:
        import yt_dlp
        from pathlib import Path
        run_ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        run_folder = Path('outputs') / f'run-{run_ts}'
        raw_inputs_dir = run_folder / 'raw_inputs'
        raw_inputs_dir.mkdir(parents=True, exist_ok=True)
        for url in args.url:
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': str(raw_inputs_dir / '%(title)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
                'noplaylist': True,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'writethumbnail': False,
                'writeinfojson': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                # Find the downloaded file
                title = info.get('title', 'downloaded')
                ext = 'wav'
                file_path = raw_inputs_dir / f"{title}.{ext}"
                url_files.append(file_path)
                url_metadata.append({
                    'source_url': url,
                    'title': title,
                    'uploader': info.get('uploader'),
                    'upload_date': info.get('upload_date'),
                    'duration': info.get('duration'),
                    'original_ext': info.get('ext'),
                    'id': info.get('id'),
                    'info_dict': info
                })
        # Set input_dir to raw_inputs_dir for downstream processing
        args.input_dir = str(raw_inputs_dir)

    # Handle --force logic
    if args.output_folder and args.resume_from and args.force:
        # Clear state and delete outputs from the specified stage forward
        from resume_utils import ResumeHelper
        run_folder = Path(args.output_folder)
        helper = ResumeHelper(run_folder)
        # Clear state
        helper.clear_from_stage(args.resume_from)
        # Delete output folders/files for all downstream stages
        stages = get_pipeline_stages()
        start_idx = stages.index(args.resume_from)
        stages_to_delete = stages[start_idx:]
        # Map stage names to output subfolders (update as needed)
        stage_to_folder = {
            'separation': 'separated',
            'normalization': 'normalized',
            'true_peak_normalization': 'true_peak_normalized',
            'clap': 'clap',
            'diarization': 'diarized',
            'segmentation': 'speakers',
            'resampling': 'speakers',
            'transcription': 'speakers',
            'soundbite_renaming': 'soundbites',
            'soundbite_finalization': 'soundbites',
            'llm': 'llm',
            'remix': 'call',
            'show': 'show',
            'finalization': 'finalized',
        }
        deleted_folders = set()
        for stage in stages_to_delete:
            folder = stage_to_folder.get(stage)
            if folder:
                target = run_folder / folder
                if target.exists():
                    import shutil
                    print(f"[FORCE] Deleting {target}")
                    shutil.rmtree(target)
                    deleted_folders.add(folder)
        print(f"[FORCE] Cleared state and deleted outputs from '{args.resume_from}' onwards.")
        # Safety check: ensure resume point has all required inputs
        # Map each stage to its required input folders
        stage_inputs = {
            'separation': ['renamed'],
            'normalization': ['separated'],
            'true_peak_normalization': ['normalized'],
            'clap': ['renamed'],
            'diarization': ['separated'],
            'segmentation': ['diarized', 'separated'],
            'resampling': ['speakers'],
            'transcription': ['speakers'],
            'soundbite_renaming': ['speakers'],
            'soundbite_finalization': ['speakers'],
            'llm': ['soundbites'],
            'remix': ['true_peak_normalized', 'normalized', 'separated'],
            'show': ['call'],
            'finalization': ['finalized', 'show', 'llm', 'soundbites'],
        }
        # Find the earliest stage >= resume_from that has all its required input folders present
        bump_stage = None
        for idx in range(start_idx, len(stages)):
            stage = stages[idx]
            required = stage_inputs.get(stage, [])
            missing = [fld for fld in required if not (run_folder / fld).exists()]
            if missing:
                bump_stage = None  # Can't run this stage
            else:
                bump_stage = stage
                break
        if bump_stage is None or bump_stage != args.resume_from:
            # Find the earliest previous stage that can be run
            for idx in range(start_idx-1, -1, -1):
                stage = stages[idx]
                required = stage_inputs.get(stage, [])
                missing = [fld for fld in required if not (run_folder / fld).exists()]
                if not missing:
                    bump_stage = stage
                    break
        if bump_stage is None:
            print(f"[ERROR] Cannot resume: no valid stage found with all required inputs present after --force deletion.")
            exit(1)
        if bump_stage != args.resume_from:
            print(f"[WARN] Cannot resume from '{args.resume_from}' because required input folders are missing after --force deletion.")
            print(f"[INFO] Auto-bumping resume point to '{bump_stage}'.")
            args.resume_from = bump_stage

    # Determine if we're resuming from an existing folder or starting fresh
    if args.output_folder:
        run_folder = Path(args.output_folder)
        if not run_folder.exists():
            print(f"❌ Output folder not found: {run_folder}")
            print(f"Available folders:")
            outputs_dir = Path("outputs")
            if outputs_dir.exists():
                run_folders = [d.name for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("run-")]
                if run_folders:
                    for folder in sorted(run_folders, reverse=True)[:5]:
                        print(f"   outputs/{folder}")
                else:
                    print(f"   No run folders found in outputs/")
            else:
                print(f"   outputs/ directory does not exist")
            exit(1)
        print(f"🔄 Resuming from existing folder: {run_folder}")

        # Handle status-only commands for specific folder
        if args.show_resume_status:
            from resume_utils import print_resume_status
            print_resume_status(run_folder, detailed=True)
            exit(0)
        if args.stage_status:
            from resume_utils import print_stage_status
            print_stage_status(run_folder, args.stage_status)
            exit(0)
        if args.clear_from:
            from resume_utils import ResumeHelper
            helper = ResumeHelper(run_folder)
            helper.clear_from_stage(args.clear_from)
            print(f"✅ Cleared completion status from '{args.clear_from}' onwards")
            exit(0)
        if args.force_rerun:
            from resume_utils import ResumeHelper
            helper = ResumeHelper(run_folder)
            helper.force_rerun_stage(args.force_rerun)
            print(f"✅ Forced stage '{args.force_rerun}' to be re-run")
            # Continue with normal run

        # Create orchestrator for existing folder - DO NOT create or reconstruct jobs
        orchestrator = PipelineOrchestrator(run_folder, args.asr_engine, args.llm_config)
        # Load manifest if it exists
        manifest_path = run_folder / 'manifest.json'
        if manifest_path.exists():
            import json
            with open(manifest_path, 'r', encoding='utf-8') as f:
                orchestrator.manifest = json.load(f)
        # No job creation or file analysis here!

    else:
        # Starting fresh run
        if not args.input_dir:
            print("❌ input_dir is required when starting a fresh run")
            print("💡 To resume an existing run, use: --output-folder outputs/run-YYYYMMDD-HHMMSS")
            exit(1)
        if args.show_resume_status:
            from resume_utils import print_resume_status
            run_folder = Path("outputs") / f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            if Path("outputs").exists():
                run_folders = [d for d in Path("outputs").iterdir() if d.is_dir() and d.name.startswith("run-")]
                if run_folders:
                    run_folder = max(run_folders, key=lambda x: x.stat().st_mtime)
            print_resume_status(run_folder, detailed=True)
            exit(0)
        if args.stage_status:
            from resume_utils import print_stage_status
            run_folder = Path("outputs") / f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            if Path("outputs").exists():
                run_folders = [d for d in Path("outputs").iterdir() if d.is_dir() and d.name.startswith("run-")]
                if run_folders:
                    run_folder = max(run_folders, key=lambda x: x.stat().st_mtime)
            print_stage_status(run_folder, args.stage_status)
            exit(0)
        input_dir = Path(args.input_dir)
        run_ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        run_folder = Path('outputs') / f'run-{run_ts}'
        print(f"🆕 Starting fresh run: {run_folder}")
        orchestrator = PipelineOrchestrator(run_folder, args.asr_engine, args.llm_config)
        jobs = create_jobs_from_input(input_dir)
        for job in jobs:
            orchestrator.add_job(job)
        if args.clear_from:
            from resume_utils import ResumeHelper
            helper = ResumeHelper(orchestrator.run_folder)
            helper.clear_from_stage(args.clear_from)
            print(f"✅ Cleared completion status from '{args.clear_from}' onwards")
            exit(0)
        if args.force_rerun:
            from resume_utils import ResumeHelper
            helper = ResumeHelper(orchestrator.run_folder)
            helper.force_rerun_stage(args.force_rerun)
            print(f"✅ Forced stage '{args.force_rerun}' to be re-run")
            # Continue with normal run
    resume_mode = args.resume or args.resume_from or args.force_rerun or args.output_folder
    resume_from_stage = args.resume_from
    if resume_mode:
        orchestrator.run_with_resume(call_tones=args.call_tones, resume=True, resume_from=resume_from_stage)
    else:
        orchestrator.run(mode=args.mode, call_cutter=args.call_cutter)
    # No redundant transcription after renaming