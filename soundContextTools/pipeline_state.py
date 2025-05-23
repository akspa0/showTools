import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import uuid


@dataclass
class StageInfo:
    """Information about a completed pipeline stage"""
    stage: str
    completed_at: str
    duration_seconds: Optional[float] = None
    jobs_processed: Optional[int] = None
    output_files: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class PipelineState:
    """
    Manages pipeline state for resume functionality.
    Tracks completed stages, current progress, and allows resuming from any point.
    Enhanced with robust error handling and recovery mechanisms.
    """
    
    def __init__(self, run_folder: Path):
        self.run_folder = Path(run_folder)
        self.state_file = self.run_folder / 'pipeline_state.json'
        self.backup_file = self.run_folder / 'pipeline_state.backup.json'
        self.state = self._load_or_create_state()
    
    def _load_or_create_state(self) -> Dict[str, Any]:
        """Load existing state or create new one with robust error handling"""
        # Try to load main state file
        if self.state_file.exists():
            try:
                state = self._load_state_file(self.state_file)
                if self._validate_state(state):
                    return state
                else:
                    print(f"⚠️  State file corrupted, attempting backup recovery...")
            except Exception as e:
                print(f"⚠️  Error loading state file: {e}")
        
        # Try backup file if main file failed
        if self.backup_file.exists():
            try:
                state = self._load_state_file(self.backup_file)
                if self._validate_state(state):
                    print(f"✅ Recovered from backup state file")
                    # Restore main file from backup
                    self._save_state(state)
                    return state
            except Exception as e:
                print(f"⚠️  Backup file also corrupted: {e}")
        
        # Create fresh state if all else fails
        print(f"🔧 Creating fresh pipeline state")
        return self._create_fresh_state()
    
    def _load_state_file(self, file_path: Path) -> Dict[str, Any]:
        """Load and parse state file with error handling"""
        try:
            content = file_path.read_text(encoding='utf-8')
            if not content.strip():
                raise ValueError("State file is empty")
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in state file: {e}")
        except Exception as e:
            raise ValueError(f"Cannot read state file: {e}")
    
    def _validate_state(self, state: Dict[str, Any]) -> bool:
        """Validate state structure and content"""
        try:
            # Check required fields
            required_fields = ['pipeline_version', 'run_id', 'created_at', 'stages']
            for field in required_fields:
                if field not in state:
                    print(f"⚠️  Missing required field: {field}")
                    return False
            
            # Validate stage entries
            if not isinstance(state['stages'], dict):
                print(f"⚠️  Invalid stages format")
                return False
            
            # Validate stage entries structure
            for stage_name, stage_info in state['stages'].items():
                if not isinstance(stage_info, dict):
                    print(f"⚠️  Invalid stage info for {stage_name}")
                    return False
                
                # Check for required stage fields
                if 'status' not in stage_info:
                    print(f"⚠️  Missing status for stage {stage_name}")
                    return False
            
            return True
        except Exception as e:
            print(f"⚠️  State validation error: {e}")
            return False
    
    def _create_fresh_state(self) -> Dict[str, Any]:
        """Create a fresh state with all stages initialized"""
        return {
            'pipeline_version': '1.0',
            'run_id': str(uuid.uuid4())[:8],
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'stages': {stage: {'status': 'pending'} for stage in get_pipeline_stages()}
        }
    
    def _save_state(self, state: Dict[str, Any] = None):
        """Save state with backup and atomic write"""
        if state is None:
            state = self.state
        
        state['updated_at'] = datetime.now().isoformat()
        
        try:
            # Ensure the run folder exists before saving
            self.run_folder.mkdir(parents=True, exist_ok=True)
            
            # Create backup of existing state
            if self.state_file.exists():
                import shutil
                shutil.copy2(self.state_file, self.backup_file)
            
            # Write to temporary file first (atomic write)
            temp_file = self.state_file.with_suffix('.tmp')
            temp_file.write_text(json.dumps(state, indent=2), encoding='utf-8')
            
            # Atomic rename (on most systems)
            temp_file.replace(self.state_file)
            
        except Exception as e:
            print(f"⚠️  Error saving state: {e}")
            # Try to restore from backup if available
            if self.backup_file.exists():
                try:
                    import shutil
                    shutil.copy2(self.backup_file, self.state_file)
                    print(f"✅ Restored state from backup")
                except Exception as restore_error:
                    print(f"❌ Failed to restore from backup: {restore_error}")
            raise
    
    def mark_stage_start(self, stage_name: str, jobs_count: Optional[int] = None):
        """Mark a stage as starting"""
        if stage_name not in self.state['stages']:
            self.state['stages'][stage_name] = {'status': 'pending'}
        
        self.state['stages'][stage_name]['status'] = 'running'
        self.state['stages'][stage_name]['started_at'] = datetime.now().isoformat()
        
        if jobs_count is not None:
            self.state['stages'][stage_name]['jobs_count'] = jobs_count
        
        self._save_state()
    
    def mark_stage_completed(self, stage: str, duration_seconds: Optional[float] = None, 
                           jobs_processed: Optional[int] = None, 
                           output_files: Optional[List[str]] = None,
                           metadata: Optional[Dict[str, Any]] = None):
        """Mark a stage as completed with enhanced error handling"""
        try:
            if stage not in self.state['stages']:
                print(f"⚠️  Unknown stage: {stage}. Adding to state.")
                self.state['stages'][stage] = {'status': 'pending'}
            
            self.state['stages'][stage] = {
                'status': 'completed',
                'completed_at': datetime.now().isoformat(),
                'duration_seconds': duration_seconds,
                'jobs_processed': jobs_processed,
                'output_files': output_files or [],
                'metadata': metadata or {}
            }
            
            # Clear any failure info for this stage
            self.state['stages'][stage].pop('failed_at', None)
            self.state['stages'][stage].pop('error_message', None)
            
            self._save_state()
            
        except Exception as e:
            print(f"❌ Error marking stage {stage} as completed: {e}")
            raise
    
    def mark_stage_failed(self, stage: str, error_message: str, 
                         duration_seconds: Optional[float] = None):
        """Mark a stage as failed with enhanced error handling"""
        try:
            if stage not in self.state['stages']:
                print(f"⚠️  Unknown stage: {stage}. Adding to state.")
                self.state['stages'][stage] = {'status': 'pending'}
            
            self.state['stages'][stage].update({
                'status': 'failed',
                'failed_at': datetime.now().isoformat(),
                'error_message': str(error_message)[:500],  # Limit error message length
                'duration_seconds': duration_seconds
            })
            
            self._save_state()
            
        except Exception as e:
            print(f"❌ Error marking stage {stage} as failed: {e}")
            # Don't re-raise here as we're already in an error state
    
    def is_stage_completed(self, stage_name: str) -> bool:
        """Check if a stage has been completed"""
        if stage_name not in self.state['stages']:
            return False
        return self.state['stages'][stage_name].get('status') == 'completed'
    
    def is_stage_failed(self, stage_name: str) -> bool:
        """Check if a stage has failed"""
        if stage_name not in self.state['stages']:
            return False
        return self.state['stages'][stage_name].get('status') == 'failed'
    
    def get_completed_stages(self) -> List[str]:
        """Get list of completed stage names"""
        return [name for name, info in self.state['stages'].items() 
                if info.get('status') == 'completed']
    
    def get_failed_stages(self) -> List[str]:
        """Get list of failed stage names"""
        return [name for name, info in self.state['stages'].items() 
                if info.get('status') == 'failed']
    
    def get_stage_info(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific stage"""
        return self.state['stages'].get(stage_name)
    
    def should_skip_stage(self, stage_name: str, force_rerun: bool = False) -> bool:
        """
        Determine if a stage should be skipped based on completion status
        
        Args:
            stage_name: Name of the stage to check
            force_rerun: If True, never skip (force re-run)
        
        Returns:
            True if stage should be skipped, False if it should run
        """
        if force_rerun:
            return False
        
        return self.is_stage_completed(stage_name)
    
    def get_next_stage_to_run(self, all_stages: List[str]) -> Optional[str]:
        """Get the next stage that needs to be run"""
        completed = set(self.get_completed_stages())
        
        for stage in all_stages:
            if stage not in completed:
                return stage
        
        return None  # All stages completed
    
    def update_job_progress(self, stage_name: str, completed_jobs: int):
        """Update the number of completed jobs for a stage"""
        if stage_name in self.state['stages']:
            self.state['stages'][stage_name]['completed_jobs'] = completed_jobs
            self._save_state()
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to the pipeline state"""
        if 'metadata' not in self.state:
            self.state['metadata'] = {}
        self.state['metadata'][key] = value
        self._save_state()
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata from the pipeline state"""
        return self.state.get('metadata', {}).get(key, default)
    
    def get_resume_summary(self) -> Dict[str, Any]:
        """Get a summary for resume operations"""
        completed = self.get_completed_stages()
        failed = self.get_failed_stages()
        
        # Find current running stage
        current_stage = None
        for name, info in self.state['stages'].items():
            if info.get('status') == 'running':
                current_stage = name
                break
        
        return {
            'run_folder': str(self.run_folder),
            'run_id': self.state.get('run_id', 'unknown'),
            'started_at': self.state.get('created_at'),
            'last_updated': self.state.get('updated_at'),
            'completed_stages': completed,
            'failed_stages': failed,
            'current_stage': current_stage,
            'total_stages_completed': len(completed),
            'has_failures': len(failed) > 0
        }
    
    def clear_stage(self, stage_name: str):
        """Clear completion status for a specific stage (force re-run)"""
        if stage_name in self.state['stages']:
            self.state['stages'][stage_name] = {'status': 'pending'}
            self._save_state()
    
    def clear_from_stage(self, stage_name: str, all_stages: List[str]):
        """Clear completion status from a specific stage onwards"""
        try:
            start_idx = all_stages.index(stage_name)
            stages_to_clear = all_stages[start_idx:]
            
            for stage in stages_to_clear:
                self.clear_stage(stage)
                
        except ValueError:
            raise ValueError(f"Stage '{stage_name}' not found in stage list")
    
    def clear_stage_from(self, stage: str):
        """Clear completion status from a stage onwards with validation"""
        try:
            stages = get_pipeline_stages()
            if stage not in stages:
                available = ', '.join(stages)
                raise ValueError(f"Unknown stage '{stage}'. Available stages: {available}")
            
            stage_index = stages.index(stage)
            stages_to_clear = stages[stage_index:]
            
            cleared_count = 0
            for stage_name in stages_to_clear:
                if stage_name in self.state['stages']:
                    if self.state['stages'][stage_name]['status'] in ['completed', 'failed']:
                        self.state['stages'][stage_name] = {'status': 'pending'}
                        cleared_count += 1
            
            self._save_state()
            print(f"✅ Cleared {cleared_count} stages from '{stage}' onwards")
            
        except Exception as e:
            print(f"❌ Error clearing stages from {stage}: {e}")
            raise
    
    def validate_stage_outputs(self, stage: str) -> bool:
        """Validate that stage outputs actually exist and are valid"""
        try:
            if not self.is_stage_completed(stage):
                return False
            
            stage_info = self.state['stages'][stage]
            output_files = stage_info.get('output_files', [])
            
            if not output_files:
                # No output files specified, can't validate
                return True
            
            # Check if all output files exist and are non-empty
            missing_files = []
            empty_files = []
            
            for file_path in output_files:
                try:
                    path = Path(file_path)
                    if not path.exists():
                        missing_files.append(str(path))
                    elif path.is_file() and path.stat().st_size == 0:
                        empty_files.append(str(path))
                except Exception as e:
                    print(f"⚠️  Error checking file {file_path}: {e}")
                    missing_files.append(str(file_path))
            
            if missing_files or empty_files:
                print(f"⚠️  Stage {stage} validation failed:")
                if missing_files:
                    print(f"   Missing files: {missing_files}")
                if empty_files:
                    print(f"   Empty files: {empty_files}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error validating stage {stage} outputs: {e}")
            return False
    
    def recover_from_partial_completion(self, stage: str) -> bool:
        """Attempt to recover from partial stage completion"""
        try:
            print(f"🔧 Attempting recovery for stage: {stage}")
            
            # Check if stage was marked completed but outputs don't exist
            if self.is_stage_completed(stage):
                if not self.validate_stage_outputs(stage):
                    print(f"🔧 Stage {stage} marked complete but outputs invalid, resetting...")
                    self.state['stages'][stage] = {'status': 'pending'}
                    self._save_state()
                    return True
            
            # Check for partial outputs that should be cleaned up
            # This could be extended based on specific stage requirements
            
            return False
            
        except Exception as e:
            print(f"❌ Error during recovery for stage {stage}: {e}")
            return False


def create_pipeline_state(run_folder: Path) -> PipelineState:
    """Factory function to create a PipelineState instance"""
    return PipelineState(run_folder)


# Common stage names for the pipeline
PIPELINE_STAGES = [
    'ingestion',
    'separation', 
    'normalization',
    'clap',
    'diarization',
    'segmentation',
    'resampling',
    'transcription',
    'soundbite_renaming',
    'soundbite_finalization',
    'llm',
    'remix',
    'show',
    'finalization'
]


def get_pipeline_stages() -> List[str]:
    """Get the standard pipeline stage names"""
    return PIPELINE_STAGES.copy() 