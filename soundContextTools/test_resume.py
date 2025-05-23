#!/usr/bin/env python3
"""
Test script for resume functionality.
Creates a minimal test environment to verify resume capabilities.
"""

import json
import tempfile
from pathlib import Path
from datetime import datetime

from pipeline_state import PipelineState, get_pipeline_stages
from resume_utils import ResumeHelper, should_resume_pipeline


def test_pipeline_state():
    """Test basic pipeline state functionality"""
    print("🧪 Testing PipelineState...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        run_folder = Path(temp_dir)
        
        # Create new state
        state = PipelineState(run_folder)
        assert state.state['pipeline_version'] == '1.0'
        assert len(state.get_completed_stages()) == 0
        
        # Mark a stage as starting
        state.mark_stage_start('ingestion', jobs_count=5)
        
        # Mark stage complete
        state.mark_stage_completed('ingestion', duration_seconds=30.5, jobs_processed=5)
        
        # Verify completion
        assert state.is_stage_completed('ingestion')
        completed = state.get_completed_stages()
        assert 'ingestion' in completed
        
        # Test failure marking
        state.mark_stage_failed('separation', 'Model failed to load')
        assert state.is_stage_failed('separation')
        
        # Test state persistence
        state2 = PipelineState(run_folder)
        assert state2.is_stage_completed('ingestion')
        assert state2.is_stage_failed('separation')
        
    print("✅ PipelineState tests passed!")


def test_resume_helper():
    """Test ResumeHelper functionality"""
    print("🧪 Testing ResumeHelper...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        run_folder = Path(temp_dir)
        helper = ResumeHelper(run_folder)
        
        # Test should_skip_stage
        assert not helper.should_skip_stage('ingestion')
        
        # Complete a stage
        helper.handle_stage_success('ingestion')
        
        # Now it should be skipped
        assert helper.should_skip_stage('ingestion')
        
        # Test failure handling
        helper.handle_stage_failure('separation', 'Test error')
        
        # Failed stages should not be skipped (will retry)
        assert not helper.should_skip_stage('separation')
        
    print("✅ ResumeHelper tests passed!")


def test_resume_detection():
    """Test resume detection logic"""
    print("🧪 Testing resume detection...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        run_folder = Path(temp_dir)
        
        # No state file = no resume
        assert not should_resume_pipeline(run_folder)
        
        # Create state with some completed stages
        state = PipelineState(run_folder)
        state.mark_stage_completed('ingestion')
        state.mark_stage_completed('separation')
        
        # Now should be resumable
        assert should_resume_pipeline(run_folder)
        
        # Complete all stages
        for stage in get_pipeline_stages():
            state.mark_stage_completed(stage)
        
        # Should not be resumable when complete
        assert not should_resume_pipeline(run_folder)
        
    print("✅ Resume detection tests passed!")


def test_state_persistence():
    """Test state file persistence and recovery"""
    print("🧪 Testing state persistence...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        run_folder = Path(temp_dir)
        
        # Create state with multiple stages
        state1 = PipelineState(run_folder)
        state1.mark_stage_completed('ingestion', duration_seconds=10.0)
        state1.mark_stage_completed('separation', jobs_processed=5)
        
        # Create new instance and verify persistence
        state2 = PipelineState(run_folder)
        assert state2.is_stage_completed('ingestion')
        assert state2.is_stage_completed('separation')
        
        # Verify state data is preserved
        ingestion_info = state2.state['stages']['ingestion']
        assert ingestion_info['duration_seconds'] == 10.0
        
        separation_info = state2.state['stages']['separation']
        assert separation_info['jobs_processed'] == 5
        
    print("✅ State persistence tests passed!")


def test_state_file_format():
    """Test state file format and JSON structure"""
    print("🧪 Testing state file format...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        run_folder = Path(temp_dir)
        state = PipelineState(run_folder)
        
        # Complete a stage to create state file
        state.mark_stage_completed('ingestion')
        
        # Read and validate JSON format
        state_file = run_folder / 'pipeline_state.json'
        assert state_file.exists()
        
        with open(state_file, 'r') as f:
            data = json.load(f)
        
        # Verify required fields
        assert 'pipeline_version' in data
        assert 'run_id' in data
        assert 'created_at' in data
        assert 'updated_at' in data
        assert 'stages' in data
        
        # Verify stage structure
        assert 'ingestion' in data['stages']
        assert data['stages']['ingestion']['status'] == 'completed'
        assert 'completed_at' in data['stages']['ingestion']
        
    print("✅ State file format tests passed!")


def demo_resume_summary():
    """Demonstrate resume summary output"""
    print("🧪 Demo: Resume Summary Output")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        run_folder = Path(temp_dir)
        helper = ResumeHelper(run_folder)
        
        # Create a realistic scenario
        helper.handle_stage_success('ingestion', duration=45.2, jobs_processed=12)
        helper.handle_stage_success('separation', duration=120.8, jobs_processed=12)
        helper.handle_stage_success('normalization', duration=15.3, jobs_processed=12)
        helper.handle_stage_success('clap', duration=95.1, jobs_processed=12)
        helper.handle_stage_success('diarization', duration=180.5, jobs_processed=12)
        
        # Simulate a failure
        helper.handle_stage_failure('transcription', 'Parakeet model failed to load')
        
        # Print summary
        from resume_utils import print_resume_status
        print_resume_status(run_folder)


if __name__ == '__main__':
    print("🚀 Testing Resume Functionality\n")
    
    test_pipeline_state()
    test_resume_helper()
    test_resume_detection()
    test_state_persistence()
    test_state_file_format()
    
    print("\n" + "="*50)
    print("🎉 All tests passed!")
    print("="*50)
    
    demo_resume_summary() 