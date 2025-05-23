import os
import shutil
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import uuid
from rich.console import Console
from rich.traceback import install as rich_traceback_install
import argparse

# Install rich traceback for better error output
rich_traceback_install()
console = Console()

# Configurable parameters
INPUT_DIR = 'input_audio'  # Change as needed
OUTPUT_DIR = 'output_audio'  # Change as needed
MANIFEST_PATH = 'manifest.json'
INDEX_PADDING = 4
SUPPORTED_EXTENSIONS = {'.wav', '.mp3', '.flac'}

# Type mapping for tuple files
TYPE_MAP = {
    'trans_out': 'right',
    'recv_out': 'left',
    'out': 'out',
}

# Logging utility
class JSONLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.log_file = open(log_path, 'a', encoding='utf-8')
    def log(self, level, stage, event, details=None):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'stage': stage,
            'event': event,
            'details': details or {}
        }
        self.log_file.write(json.dumps(entry) + '\n')
        self.log_file.flush()
        # Console output
        if level == 'ERROR':
            console.print(f"[bold red]{level}[/] [{stage}] {event}: {details}")
        elif level == 'WARNING':
            console.print(f"[yellow]{level}[/] [{stage}] {event}: {details}")
        else:
            console.print(f"[green]{level}[/] [{stage}] {event}: {details}")
    def close(self):
        self.log_file.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Audio Context Tool File Input Stage")
    parser.add_argument('input_dir', type=str, help='Input directory')
    return parser.parse_args()

def remove_pii(filename):
    """
    Placeholder for PII removal logic. Replace with actual implementation.
    For now, just returns the filename unchanged.
    """
    # TODO: Implement real PII removal
    return filename

def get_chronological_index(ts, used_indices):
    """
    Assigns a unique, zero-padded index based on timestamp.
    """
    idx = 0
    while True:
        index_str = f"{idx:0{INDEX_PADDING}d}"
        if index_str not in used_indices:
            used_indices.add(index_str)
            return index_str
        idx += 1

def extract_type(filename):
    for key, mapped in TYPE_MAP.items():
        if key in filename:
            return mapped, key
    return None, None

def get_file_timestamp(path):
    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts)

def group_files_by_tuple(files):
    """
    Group files by (timestamp, tuple_id) for tuple renaming.
    Tuple_id is based on timestamp rounded to the second and tuple prefix.
    """
    tuples = defaultdict(list)
    singles = []
    for orig_path, file in files:
        ext = Path(file).suffix.lower()
        file_type, tuple_key = extract_type(file)
        ts = get_file_timestamp(orig_path)
        ts_str = ts.strftime('%Y%m%d-%H%M%S')
        if file_type:
            tuple_id = (ts_str, file_type)
            tuples[tuple_id].append((orig_path, file, file_type, ts))
        else:
            singles.append((orig_path, file, ts))
    return tuples, singles

def scan_and_ingest(input_dir, output_dir, manifest_path):
    manifest = []
    used_indices = set()
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        rel_root = Path(root).relative_to(input_dir)
        for file in files:
            ext = Path(file).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            orig_path = Path(root) / file
            # Remove PII from filename
            clean_name = remove_pii(file)
            # Use file's modification time for chronological index
            ts = os.path.getmtime(orig_path)
            index = get_chronological_index(ts, used_indices)
            new_name = f"{index}-{clean_name}"
            rel_output_dir = output_dir / rel_root
            rel_output_dir.mkdir(parents=True, exist_ok=True)
            dest_path = rel_output_dir / new_name
            shutil.copy2(orig_path, dest_path)
            # Add to manifest
            manifest.append({
                'index': index,
                'original_name': file,
                'clean_name': clean_name,
                'input_path': str(orig_path),
                'output_path': str(dest_path),
                'timestamp': datetime.fromtimestamp(ts).isoformat(),
            })
    # Write manifest
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f"Ingestion complete. Manifest written to {manifest_path}.")

def ingest_and_rename(input_dir, output_dir, manifest_path, show_input=False, show_output=False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    tuples, singles = group_files_by_tuple(input_dir)
    used_indices = set()
    index = 0
    # Process tuples
    for tuple_id, files in sorted(tuples.items()):
        index_str = f"{index:0{INDEX_PADDING}d}"
        ts_str = tuple_id[0]
        for orig_path, orig_name, file_type, ts in files:
            new_name = f"{index_str}-{file_type}-{ts_str}{Path(orig_name).suffix.lower()}"
            rel_output_dir = output_dir
            rel_output_dir.mkdir(parents=True, exist_ok=True)
            dest_path = rel_output_dir / new_name
            shutil.copy2(orig_path, dest_path)
            manifest.append({
                'index': index_str,
                'tuple_id': tuple_id,
                'type': file_type,
                'timestamp': ts_str,
                'original_name': orig_name,
                'output_name': new_name,
                'input_path': str(orig_path),
                'output_path': str(dest_path),
            })
        index += 1
    # Process singles
    for orig_path, orig_name, ts in singles:
        index_str = f"{index:0{INDEX_PADDING}d}"
        ts_str = ts.strftime('%Y%m%d-%H%M%S')
        new_name = f"{index_str}-single-{ts_str}{Path(orig_name).suffix.lower()}"
        rel_output_dir = output_dir
        rel_output_dir.mkdir(parents=True, exist_ok=True)
        dest_path = rel_output_dir / new_name
        shutil.copy2(orig_path, dest_path)
        manifest.append({
            'index': index_str,
            'tuple_id': None,
            'type': 'single',
            'timestamp': ts_str,
            'original_name': orig_name,
            'output_name': new_name,
            'input_path': str(orig_path),
            'output_path': str(dest_path),
        })
        index += 1
    # Write manifest
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump({
            'show_input': show_input,
            'show_output': show_output,
            'files': manifest
        }, f, indent=2)
    print(f"Ingestion and renaming complete. Manifest written to {manifest_path}.")

def redact_pii(details, is_tuple):
    if not is_tuple:
        return details
    # Remove or censor any PII/original filename/path fields
    redacted = {}
    for k, v in details.items():
        if k in ('from', 'orig_path', 'original_name'):
            continue  # Remove PII fields
        redacted[k] = v
    return redacted

def process_file_job(job, run_folder: Path):
    """
    Process a single file/tuple as a job: copy to raw_inputs, rename if needed, and return result dict.
    job.data must contain at least: 'orig_path', 'base_name', 'is_tuple', 'tuple_index', 'file_type', 'timestamp', 'ext'
    Returns: {'success': bool, 'output_name': str, 'output_path': str, 'error': str or None}
    """
    raw_inputs_dir = run_folder / 'raw_inputs'
    renamed_dir = run_folder / 'renamed'
    raw_inputs_dir.mkdir(parents=True, exist_ok=True)
    renamed_dir.mkdir(parents=True, exist_ok=True)
    orig_path = Path(job.data['orig_path'])
    base_name = job.data['base_name']
    ext = job.data['ext']
    # Copy to raw_inputs (flat), but do NOT log anything
    dest_raw = raw_inputs_dir / base_name
    try:
        shutil.copy2(orig_path, dest_raw)
    except Exception as e:
        return {'success': False, 'output_name': None, 'output_path': None, 'error': str(e)}
    # Rename and copy to renamed/
    if job.data['is_tuple']:
        index_str = f"{job.data['tuple_index']:0{INDEX_PADDING}d}"
        ts_str = job.data['timestamp']
        file_type = job.data['file_type']
        new_name = f"{index_str}-{file_type}-{ts_str}{ext}"
        dest_renamed = renamed_dir / new_name
        try:
            shutil.copy2(dest_raw, dest_renamed)
            # Delete the file from raw_inputs after successful rename
            try:
                dest_raw.unlink()
            except Exception as del_err:
                return {'success': False, 'output_name': new_name, 'output_path': str(dest_renamed), 'error': f'File renamed but failed to delete raw_inputs file: {del_err}'}
            job.data['output_name'] = new_name
            job.data['output_path'] = str(dest_renamed)
            job.data['stage'] = 'renamed'
            return {'success': True, 'output_name': new_name, 'output_path': str(dest_renamed), 'error': None}
        except Exception as e:
            return {'success': False, 'output_name': new_name, 'output_path': str(dest_renamed), 'error': str(e)}
    else:
        # Non-tuple file: apply same privacy rule, only log after renaming
        index_str = f"{job.data['tuple_index']:0{INDEX_PADDING}d}"
        ts_str = job.data['timestamp']
        new_name = f"{index_str}-single-{ts_str}{ext}"
        dest_renamed = renamed_dir / new_name
        try:
            shutil.copy2(dest_raw, dest_renamed)
            # Delete the file from raw_inputs after successful rename
            try:
                dest_raw.unlink()
            except Exception as del_err:
                return {'success': False, 'output_name': new_name, 'output_path': str(dest_renamed), 'error': f'File renamed but failed to delete raw_inputs file: {del_err}'}
            job.data['output_name'] = new_name
            job.data['output_path'] = str(dest_renamed)
            job.data['stage'] = 'renamed'
            return {'success': True, 'output_name': new_name, 'output_path': str(dest_renamed), 'error': None}
        except Exception as e:
            return {'success': False, 'output_name': new_name, 'output_path': str(dest_renamed), 'error': str(e)}

def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    # Timestamped run folder
    run_ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_folder = Path('output') / run_ts
    run_folder.mkdir(parents=True, exist_ok=True)
    # Set up subfolders
    raw_inputs_dir = run_folder / 'raw_inputs'
    renamed_dir = run_folder / 'renamed'
    raw_inputs_dir.mkdir(parents=True, exist_ok=True)
    renamed_dir.mkdir(parents=True, exist_ok=True)
    # Set up logging
    log_path = run_folder / f'run-{run_ts}.log.json'
    logger = JSONLogger(log_path)
    manifest_path = run_folder / 'manifest.json'
    manifest = []
    # Copy all input files flat to raw_inputs
    all_files = []
    seen_names = set()
    for root, _, files in os.walk(input_dir):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            orig_path = Path(root) / file
            # Ensure unique filename in raw_inputs
            base_name = Path(file).name
            if base_name in seen_names:
                base_name = f"{Path(file).stem}_{uuid.uuid4().hex[:6]}{Path(file).suffix}"
            seen_names.add(base_name)
            dest_path = raw_inputs_dir / base_name
            try:
                shutil.copy2(orig_path, dest_path)
                # DO NOT log original paths or names (PII) at this stage
                all_files.append((dest_path, base_name))
            except Exception as e:
                # Only log anonymized error (no original path)
                logger.log('ERROR', 'raw_inputs', 'file_copy_failed', {'error': str(e)})
    # Group files for renaming
    tuples, singles = group_files_by_tuple(all_files)
    used_indices = set()
    index = 0
    # Process tuples (call files)
    for tuple_id, files in sorted(tuples.items()):
        index_str = f"{index:0{INDEX_PADDING}d}"
        ts_str = tuple_id[0]
        for orig_path, orig_name, file_type, ts in files:
            new_name = f"{index_str}-{file_type}-{ts_str}{Path(orig_name).suffix.lower()}"
            dest_path = renamed_dir / new_name
            try:
                shutil.copy2(orig_path, dest_path)
                # Delete the file from raw_inputs after successful rename
                try:
                    Path(orig_path).unlink()
                except Exception as del_err:
                    logger.log('ERROR', 'renamed', 'raw_input_delete_failed', {'output_name': new_name, 'output_path': str(dest_path), 'error': str(del_err)})
                # Only log anonymized fields after PII is deleted
                logger.log('INFO', 'renamed', 'file_renamed', {'output_name': new_name, 'output_path': str(dest_path)})
                # Manifest for call tuples: NO original filename
                manifest.append({
                    'index': index_str,
                    'type': file_type,
                    'timestamp': ts_str,
                    'output_name': new_name,
                    'output_path': str(dest_path),
                    'stage': 'renamed',
                })
            except Exception as e:
                logger.log('ERROR', 'renamed', 'file_rename_failed', {'output_name': new_name, 'output_path': str(dest_path), 'error': str(e)})
        index += 1
    # Process singles (non-tuple files)
    for orig_path, orig_name, ts in singles:
        index_str = f"{index:0{INDEX_PADDING}d}"
        ts_str = ts.strftime('%Y%m%d-%H%M%S')
        new_name = f"{index_str}-single-{ts_str}{Path(orig_name).suffix.lower()}"
        dest_path = renamed_dir / new_name
        try:
            shutil.copy2(orig_path, dest_path)
            # Delete the file from raw_inputs after successful rename
            try:
                Path(orig_path).unlink()
            except Exception as del_err:
                logger.log('ERROR', 'renamed', 'raw_input_delete_failed', {'output_name': new_name, 'output_path': str(dest_path), 'error': str(del_err)})
            # Only log anonymized fields after PII is deleted
            logger.log('INFO', 'renamed', 'file_renamed', {'output_name': new_name, 'output_path': str(dest_path)})
            # Manifest for singles: NO original filename
            manifest.append({
                'index': index_str,
                'type': 'single',
                'timestamp': ts_str,
                'output_name': new_name,
                'output_path': str(dest_path),
                'stage': 'renamed',
            })
        except Exception as e:
            logger.log('ERROR', 'renamed', 'file_rename_failed', {'output_name': new_name, 'output_path': str(dest_path), 'error': str(e)})
        index += 1
    # Write manifest
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    logger.log('INFO', 'pipeline', 'manifest_written', {'manifest_path': str(manifest_path)})
    logger.close()
    console.print(f"[bold green]File input stage complete. Outputs in {run_folder}[/]")

if __name__ == '__main__':
    main() 