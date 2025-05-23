import time
import functools
from pathlib import Path
from typing import Callable, Any, Optional, List, Dict
from pipeline_state import PipelineState, get_pipeline_stages


def with_resume_support(stage_name: str, pipeline_state: Optional[PipelineState] = None):
    """
    Decorator to add resume functionality to any pipeline stage method.
    
    Usage:
        @with_resume_support('separation', orchestrator.pipeline_state)
        def run_audio_separation_stage(self):
            # existing logic unchanged
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the orchestrator instance (assumes it's the first argument)
            orchestrator = args[0] if args else None
            
            # Use provided pipeline_state or try to get from orchestrator
            state = pipeline_state
            if state is None and hasattr(orchestrator, 'pipeline_state'):
                state = orchestrator.pipeline_state
            
            # If no state management, just run normally
            if state is None or not getattr(orchestrator, 'resume_mode', False):
                return func(*args, **kwargs)
            
            # Check if we should skip this stage
            if state.should_skip_stage(stage_name):
                if hasattr(orchestrator, 'log_event'):
                    orchestrator.log_event('INFO', f'{stage_name}_skipped', 
                                         {'reason': 'already_completed'})
                return None
            
            # Mark stage start and run with timing
            start_time = time.time()
            state.mark_stage_start(stage_name)
            
            if hasattr(orchestrator, 'log_event'):
                orchestrator.log_event('INFO', f'{stage_name}_resume_start')
            
            try:
                result = func(*args, **kwargs)
                
                # Mark stage complete
                duration = time.time() - start_time
                state.mark_stage_complete(stage_name, duration_seconds=duration)
                
                if hasattr(orchestrator, 'log_event'):
                    orchestrator.log_event('INFO', f'{stage_name}_resume_complete', 
                                         {'duration_seconds': duration})
                
                return result
                
            except Exception as e:
                # Mark stage failed
                state.mark_stage_failed(stage_name, str(e))
                
                if hasattr(orchestrator, 'log_event'):
                    orchestrator.log_event('ERROR', f'{stage_name}_resume_failed', 
                                         {'error': str(e)})
                raise
        
        return wrapper
    return decorator


class ResumeHelper:
    """Enhanced helper for resume functionality with robust error handling"""
    
    def __init__(self, run_folder: Path):
        self.run_folder = Path(run_folder)
        self.pipeline_state = PipelineState(run_folder)
    
    def should_skip_stage(self, stage: str, validate_outputs: bool = True) -> bool:
        """
        Determine if a stage should be skipped with output validation.
        
        Args:
            stage: Stage name to check
            validate_outputs: Whether to validate stage outputs exist
        """
        try:
            if not self.pipeline_state.is_stage_completed(stage):
                return False
            
            if validate_outputs:
                if not self.pipeline_state.validate_stage_outputs(stage):
                    print(f"⚠️  Stage {stage} marked complete but outputs invalid, will re-run")
                    # Automatically reset the stage
                    self.pipeline_state.recover_from_partial_completion(stage)
                    return False
            
            print(f"✅ Skipping completed stage: {stage}")
            return True
            
        except Exception as e:
            print(f"❌ Error checking stage {stage}: {e}")
            # In case of error, don't skip (safer to re-run)
            return False
    
    def prepare_stage_run(self, stage: str) -> bool:
        """
        Prepare for running a stage, including recovery attempts.
        
        Returns:
            True if stage should be run, False if should be skipped
        """
        try:
            # Check if stage should be skipped
            if self.should_skip_stage(stage):
                return False
            
            # Attempt recovery if stage was failed
            if self.pipeline_state.is_stage_failed(stage):
                print(f"🔧 Stage {stage} previously failed, attempting recovery...")
                if self.pipeline_state.recover_from_partial_completion(stage):
                    print(f"✅ Recovery successful for {stage}")
                else:
                    print(f"⚠️  Recovery could not fix {stage}, will retry anyway")
            
            # Mark stage as starting
            self.pipeline_state.mark_stage_start(stage)
            return True
            
        except Exception as e:
            print(f"❌ Error preparing stage {stage}: {e}")
            # In case of error, try to run anyway
            return True
    
    def handle_stage_success(self, stage: str, duration: Optional[float] = None,
                           jobs_processed: Optional[int] = None,
                           output_files: Optional[List[str]] = None,
                           metadata: Optional[Dict[str, Any]] = None):
        """Handle successful stage completion with validation"""
        try:
            self.pipeline_state.mark_stage_completed(
                stage, duration, jobs_processed, output_files, metadata
            )
            print(f"✅ Stage {stage} completed successfully")
            
            # Validate outputs were actually created
            if output_files and not self.pipeline_state.validate_stage_outputs(stage):
                print(f"⚠️  Warning: Stage {stage} completed but output validation failed")
            
        except Exception as e:
            print(f"❌ Error recording stage {stage} success: {e}")
    
    def handle_stage_failure(self, stage: str, error: str, duration: Optional[float] = None):
        """Handle stage failure with enhanced error info"""
        try:
            self.pipeline_state.mark_stage_failed(stage, str(error), duration)
            print(f"❌ Stage {stage} failed: {error}")
            
            # Attempt automatic recovery for certain error types
            if self._is_recoverable_error(error):
                print(f"🔧 Attempting automatic recovery for {stage}...")
                if self.pipeline_state.recover_from_partial_completion(stage):
                    print(f"✅ Automatic recovery successful for {stage}")
                    return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error recording stage {stage} failure: {e}")
            return False
    
    def _is_recoverable_error(self, error: str) -> bool:
        """Check if an error might be recoverable through cleanup"""
        recoverable_patterns = [
            'file not found',
            'no such file',
            'empty file',
            'corrupted',
            'incomplete',
            'permission denied'
        ]
        
        error_lower = error.lower()
        return any(pattern in error_lower for pattern in recoverable_patterns)
    
    def force_rerun_stage(self, stage: str):
        """Force a stage to be re-run by clearing its completion status"""
        try:
            if stage in self.pipeline_state.state['stages']:
                self.pipeline_state.state['stages'][stage] = {'status': 'pending'}
                self.pipeline_state._save_state()
                print(f"🔧 Forced stage {stage} to be re-run")
            else:
                print(f"⚠️  Stage {stage} not found in state")
        except Exception as e:
            print(f"❌ Error forcing rerun of stage {stage}: {e}")
    
    def clear_from_stage(self, stage: str):
        """Clear completion status from a stage onwards"""
        try:
            self.pipeline_state.clear_stage_from(stage)
        except Exception as e:
            print(f"❌ Error clearing from stage {stage}: {e}")
            raise
    
    def get_resume_summary(self) -> Dict[str, Any]:
        """Get detailed resume summary with enhanced info"""
        try:
            state = self.pipeline_state.state
            stages = self.pipeline_state.state.get('stages', {})
            
            completed_stages = [name for name, info in stages.items() 
                              if info.get('status') == 'completed']
            failed_stages = [name for name, info in stages.items() 
                           if info.get('status') == 'failed']
            pending_stages = [name for name, info in stages.items() 
                            if info.get('status') == 'pending']
            
            # Find next stage to run
            all_stages = get_pipeline_stages()
            next_stage = None
            for stage in all_stages:
                if stage in pending_stages:
                    next_stage = stage
                    break
            
            return {
                'run_folder': str(self.run_folder),
                'run_id': state.get('run_id', 'unknown'),
                'created_at': state.get('created_at'),
                'updated_at': state.get('updated_at'),
                'completed_stages': completed_stages,
                'failed_stages': failed_stages,
                'pending_stages': pending_stages,
                'next_stage': next_stage,
                'total_stages': len(all_stages),
                'completion_percentage': len(completed_stages) / len(all_stages) * 100 if all_stages else 0
            }
            
        except Exception as e:
            print(f"❌ Error generating resume summary: {e}")
            return {'error': str(e)}


def add_resume_to_orchestrator(orchestrator, resume_mode: bool = False):
    """
    Add resume functionality to an existing orchestrator instance.
    This is a non-breaking enhancement that can be called optionally.
    
    Usage:
        orchestrator = PipelineOrchestrator(run_folder)
        add_resume_to_orchestrator(orchestrator, resume_mode=True)
        # Now orchestrator has resume capabilities
    """
    # Add resume helper
    orchestrator.resume_helper = ResumeHelper(orchestrator.run_folder)
    orchestrator.resume_mode = resume_mode
    orchestrator.pipeline_state = orchestrator.resume_helper.pipeline_state
    
    # Add convenience methods
    orchestrator.should_skip_stage = orchestrator.resume_helper.should_skip_stage
    orchestrator.prepare_stage_run = orchestrator.resume_helper.prepare_stage_run
    orchestrator.handle_stage_success = orchestrator.resume_helper.handle_stage_success
    orchestrator.handle_stage_failure = orchestrator.resume_helper.handle_stage_failure
    orchestrator.force_rerun_stage = orchestrator.resume_helper.force_rerun_stage
    orchestrator.clear_from_stage = orchestrator.resume_helper.clear_from_stage
    orchestrator.get_resume_summary = orchestrator.resume_helper.get_resume_summary
    
    return orchestrator


def print_stage_status(run_folder: Path, stage_name: str):
    """Print detailed status for a specific stage"""
    try:
        if not run_folder.exists():
            print(f"❌ Run folder does not exist: {run_folder}")
            return
        
        helper = ResumeHelper(run_folder)
        stage_info = helper.pipeline_state.get_stage_info(stage_name)
        
        if not stage_info:
            available_stages = ', '.join(get_pipeline_stages())
            print(f"❌ Stage '{stage_name}' not found.")
            print(f"Available stages: {available_stages}")
            return
        
        print(f"\n" + "="*50)
        print(f"📊 STAGE STATUS: {stage_name.upper()}")
        print("="*50)
        
        status = stage_info.get('status', 'unknown')
        status_emoji = {
            'pending': '⏳',
            'running': '🔄', 
            'completed': '✅',
            'failed': '❌'
        }.get(status, '❓')
        
        print(f"Status: {status_emoji} {status.upper()}")
        
        if status == 'completed':
            completed_at = stage_info.get('completed_at', 'Unknown')
            duration = stage_info.get('duration_seconds')
            jobs_processed = stage_info.get('jobs_processed')
            output_files = stage_info.get('output_files', [])
            
            print(f"✅ Completed: {completed_at}")
            if duration:
                print(f"⏱️  Duration: {duration:.1f} seconds")
            if jobs_processed:
                print(f"📊 Jobs Processed: {jobs_processed}")
            if output_files:
                print(f"📁 Output Files ({len(output_files)}):")
                for i, file_path in enumerate(output_files[:5]):  # Show first 5
                    print(f"   {i+1}. {file_path}")
                if len(output_files) > 5:
                    print(f"   ... and {len(output_files) - 5} more")
        
        elif status == 'failed':
            failed_at = stage_info.get('failed_at', 'Unknown')
            error_message = stage_info.get('error_message', 'Unknown error')
            duration = stage_info.get('duration_seconds')
            
            print(f"❌ Failed: {failed_at}")
            print(f"💥 Error: {error_message}")
            if duration:
                print(f"⏱️  Duration: {duration:.1f} seconds")
        
        elif status == 'running':
            started_at = stage_info.get('started_at', 'Unknown')
            jobs_count = stage_info.get('jobs_count')
            
            print(f"🔄 Started: {started_at}")
            if jobs_count:
                print(f"📊 Total Jobs: {jobs_count}")
        
        elif status == 'pending':
            print(f"⏳ Waiting to be executed")
        
        # Show metadata if available
        metadata = stage_info.get('metadata', {})
        if metadata:
            print(f"\n📋 Metadata:")
            for key, value in metadata.items():
                print(f"   {key}: {value}")
        
        print("="*50)
        
    except Exception as e:
        print(f"❌ Error getting stage status: {e}")


def run_stage_with_resume(orchestrator, stage_name: str, stage_func: Callable, resume_from: str = None, *args, **kwargs):
    """
    Run a stage with resume support and advanced controls.
    
    Args:
        orchestrator: The pipeline orchestrator instance
        stage_name: Name of the stage to run
        stage_func: Function to execute for this stage  
        resume_from: If specified, only run stages from this point onwards
        *args, **kwargs: Arguments to pass to stage_func
    """
    # If resume is not enabled, just run normally
    if not getattr(orchestrator, 'resume_mode', False):
        return stage_func(*args, **kwargs)
    
    # If resume_from is specified, check if we should skip this stage
    if resume_from:
        all_stages = get_pipeline_stages()
        if resume_from not in all_stages:
            available = ', '.join(all_stages)
            raise ValueError(f"Invalid resume_from stage '{resume_from}'. Available: {available}")
        
        # Skip stages before the resume_from point
        resume_from_index = all_stages.index(resume_from)
        current_stage_index = all_stages.index(stage_name) if stage_name in all_stages else -1
        
        if current_stage_index >= 0 and current_stage_index < resume_from_index:
            print(f"⏭️  Skipping stage '{stage_name}' (before resume point '{resume_from}')")
            orchestrator.log_event('INFO', f'{stage_name}_skipped', 
                                 {'reason': f'before_resume_point_{resume_from}'})
            return None
    
    # Check if we should skip this stage normally
    if orchestrator.should_skip_stage(stage_name):
        orchestrator.log_event('INFO', f'{stage_name}_skipped', {'reason': 'already_completed'})
        return None
    
    # Run with timing and state tracking
    start_time = time.time()
    if not orchestrator.prepare_stage_run(stage_name):
        return None  # Stage was skipped during preparation
    
    orchestrator.log_event('INFO', f'{stage_name}_resume_start')
    
    try:
        result = stage_func(*args, **kwargs)
        
        duration = time.time() - start_time
        orchestrator.handle_stage_success(stage_name, duration=duration)
        orchestrator.log_event('INFO', f'{stage_name}_resume_complete', 
                             {'duration_seconds': duration})
        
        return result
        
    except Exception as e:
        duration = time.time() - start_time
        orchestrator.handle_stage_failure(stage_name, str(e), duration)
        orchestrator.log_event('ERROR', f'{stage_name}_resume_failed', {'error': str(e)})
        raise


def create_resume_enabled_run_method(original_run_method: Callable) -> Callable:
    """
    Create a resume-enabled version of the orchestrator's run method.
    This preserves the original method while adding resume capabilities.
    
    Usage:
        orchestrator.run_with_resume = create_resume_enabled_run_method(orchestrator.run)
    """
    @functools.wraps(original_run_method)
    def run_with_resume(self, call_tones=False, resume=False):
        # If not resuming, use original method
        if not resume:
            return original_run_method(self, call_tones=call_tones)
        
        # Enable resume mode
        add_resume_to_orchestrator(self, resume_mode=True)
        
        # Print resume summary
        print_resume_status(self.run_folder, detailed=True)
        
        # Run stages with resume support
        # Note: This would need to be customized based on the actual stage methods
        # For now, we'll use the stage wrapper approach
        
        stages_and_methods = [
            ('ingestion', lambda: self._run_ingestion_jobs()),
            ('separation', lambda: (self.add_separation_jobs(), self.run_audio_separation_stage())),
            ('normalization', self.run_normalization_stage),
            ('clap', self.run_clap_annotation_stage),
            ('diarization', self.run_diarization_stage),
            ('segmentation', self.run_speaker_segmentation_stage),
            ('resampling', self.run_resample_segments_stage),
            ('transcription', lambda: self.run_transcription_stage(asr_engine='parakeet')),
            ('soundbite_renaming', self.run_rename_soundbites_stage),
            ('soundbite_finalization', self.run_final_soundbite_stage),
            ('llm', self.run_llm_stage),
            ('remix', lambda: self.run_remix_stage(call_tones=call_tones)),
            ('show', lambda: self.run_show_stage(call_tones=call_tones))
        ]
        
        for stage_name, stage_method in stages_and_methods:
            run_stage_with_resume(self, stage_name, stage_method)
        
        # Final summary
        print_resume_status(self.run_folder, detailed=True)
    
    return run_with_resume


def should_resume_pipeline(run_folder: Path, stage_name: str = None) -> bool:
    """
    Enhanced check if pipeline should resume with better error handling.
    
    Args:
        run_folder: Path to run folder
        stage_name: Specific stage to check (optional)
    """
    try:
        if not run_folder.exists():
            return False
        
        state_file = run_folder / 'pipeline_state.json'
        backup_file = run_folder / 'pipeline_state.backup.json'
        
        # Check if any state file exists
        if not state_file.exists() and not backup_file.exists():
            return False
        
        # Try to load state with error handling
        try:
            pipeline_state = PipelineState(run_folder)
            
            # If specific stage requested, check if it's incomplete
            if stage_name:
                return not pipeline_state.is_stage_completed(stage_name)
            
            # Otherwise check if any stage is completed (indicating partial run)
            stages = pipeline_state.state.get('stages', {})
            has_completed = any(info.get('status') == 'completed' for info in stages.values())
            has_pending = any(info.get('status') == 'pending' for info in stages.values())
            
            return has_completed and has_pending
            
        except Exception as e:
            print(f"⚠️  Error checking resume state: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Error in should_resume_pipeline: {e}")
        return False


def print_resume_status(run_folder: Path, detailed: bool = False):
    """Print enhanced resume status with error handling"""
    try:
        if not should_resume_pipeline(run_folder):
            print("❌ No resumable pipeline found in this folder")
            return
        
        helper = ResumeHelper(run_folder)
        summary = helper.get_resume_summary()
        
        if 'error' in summary:
            print(f"❌ Error getting resume status: {summary['error']}")
            return
        
        print("\n" + "="*50)
        print("🔄 PIPELINE RESUME STATUS")
        print("="*50)
        print(f"📁 Run Folder: {summary['run_folder']}")
        print(f"🆔 Run ID: {summary['run_id']}")
        print(f"📅 Started: {summary.get('created_at', 'Unknown')}")
        print(f"🔄 Last Updated: {summary.get('updated_at', 'Unknown')}")
        print(f"📊 Progress: {summary['completion_percentage']:.1f}% complete")
        print(f"   ({len(summary['completed_stages'])}/{summary['total_stages']} stages)")
        
        if summary['completed_stages']:
            print(f"\n✅ Completed Stages ({len(summary['completed_stages'])}):")
            for stage in summary['completed_stages']:
                print(f"   ✓ {stage}")
        
        if summary['failed_stages']:
            print(f"\n❌ Failed Stages ({len(summary['failed_stages'])}):")
            for stage in summary['failed_stages']:
                stage_info = helper.pipeline_state.state['stages'][stage]
                error = stage_info.get('error_message', 'Unknown error')
                failed_at = stage_info.get('failed_at', 'Unknown time')
                print(f"   ✗ {stage}")
                print(f"     Error: {error}")
                print(f"     Failed at: {failed_at}")
        
        if summary['next_stage']:
            print(f"\n🎯 Next Stage: {summary['next_stage']}")
        
        if summary['pending_stages']:
            remaining = len(summary['pending_stages'])
            print(f"\n⏳ Remaining Stages ({remaining}):")
            for stage in summary['pending_stages'][:5]:  # Show first 5
                print(f"   ○ {stage}")
            if remaining > 5:
                print(f"   ... and {remaining - 5} more")
        
        if detailed:
            print(f"\n📂 Output Structure:")
            try:
                for item in sorted(run_folder.iterdir()):
                    if item.is_dir():
                        file_count = len(list(item.glob('*')))
                        print(f"   📁 {item.name}/ ({file_count} files)")
            except Exception as e:
                print(f"   ❌ Error reading output structure: {e}")
        
        print("="*50)
        
    except Exception as e:
        print(f"❌ Error printing resume status: {e}") 