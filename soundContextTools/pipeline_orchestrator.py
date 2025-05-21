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
        """
        renamed_dir = self.run_folder / 'renamed'
        clap_dir = self.run_folder / 'clap'
        prompts = [
            'dog barking', 'DTMF', 'ringing', 'yelling', 'music', 'laughter', 'crying', 'doorbell', 'car horn', 'applause', 'gunshot', 'siren', 'footsteps', 'phone hangup', 'phone pickup', 'busy signal', 'static', 'noise', 'speech', 'silence'
        ]
        self.log_event('INFO', 'clap_annotation_start', {'prompts': prompts})
        results = annotate_clap_for_out_files(renamed_dir, clap_dir, prompts)
        for result in results:
            self.log_event('INFO', 'clap_annotation_result', {
                'call_id': result['call_id'],
                'input_name': result['input_name'],
                'annotation_path': result['annotation_path'],
                'accepted_annotations': result['accepted_annotations']
            })
            self.manifest.append({
                'stage': 'clap',
                'call_id': result['call_id'],
                'input_name': result['input_name'],
                'annotation_path': result['annotation_path'],
                'accepted_annotations': result['accepted_annotations']
            })
        self.log_event('INFO', 'clap_annotation_complete', {'count': len(results)})

    def run(self):
        # Buffer all logs/console output until after all jobs are processed and raw_inputs are deleted
        with tqdm(total=len(self.jobs), desc="Global Progress", position=0) as global_bar:
            for job in self.jobs:
                if job.job_type == 'rename':
                    result = process_file_job(job, self.run_folder)
                    # Only use anonymized fields from result and job.data for logging/manifest
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
        # At this point, all jobs are processed and raw_inputs should be deleted by process_file_job
        # Now it is safe to enable logging and flush buffered console output
        self.enable_logging()
        self.write_log()
        self.write_manifest()
        # Add and run audio separation jobs
        self.add_separation_jobs()
        self.run_audio_separation_stage()
        # Run CLAP annotation stage
        self.run_clap_annotation_stage()

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
    orchestrator.run() 