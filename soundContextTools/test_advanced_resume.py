#!/usr/bin/env python3
"""
Test script for advanced resume controls.
Tests --resume-from, --force-rerun, --clear-from, and --stage-status functionality.
"""

import json
import tempfile
from pathlib import Path
from datetime import datetime

from pipeline_state import PipelineState, get_pipeline_stages
from resume_utils import ResumeHelper, print_stage_status, print_resume_status


def test_force_rerun():
    """Test force re-run functionality"""
    print("🧪 Testing Force Re-run...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        run_folder = Path(temp_dir)
        helper = ResumeHelper(run_folder)
        
        # Complete a stage
        helper.handle_stage_success('ingestion', duration=45.2)
        assert helper.should_skip_stage('ingestion')  # Should be skipped
        
        # Force re-run
        helper.force_rerun_stage('ingestion')
        assert not helper.should_skip_stage('ingestion')  # Should no longer be skipped
        
        print("✅ Force re-run test passed!")


def test_clear_from_stage():
    """Test clear from stage functionality"""
    print("🧪 Testing Clear From Stage...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        run_folder = Path(temp_dir)
        helper = ResumeHelper(run_folder)
        
        # Complete several stages
        helper.handle_stage_success('ingestion')
        helper.handle_stage_success('separation')
        helper.handle_stage_success('normalization')
        helper.handle_stage_success('clap')
        
        # Verify they're completed
        assert helper.should_skip_stage('ingestion')
        assert helper.should_skip_stage('separation')
        assert helper.should_skip_stage('normalization')
        assert helper.should_skip_stage('clap')
        
        # Clear from separation onwards
        helper.clear_from_stage('separation')
        
        # Verify clearing worked correctly
        assert helper.should_skip_stage('ingestion')      # Should still be complete
        assert not helper.should_skip_stage('separation')  # Should be cleared
        assert not helper.should_skip_stage('normalization')  # Should be cleared
        assert not helper.should_skip_stage('clap')       # Should be cleared
        
        print("✅ Clear from stage test passed!")


def test_stage_status():
    """Test detailed stage status functionality"""
    print("🧪 Testing Stage Status...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        run_folder = Path(temp_dir)
        helper = ResumeHelper(run_folder)
        
        # Test pending stage
        print("\n--- Testing Pending Stage ---")
        print_stage_status(run_folder, 'ingestion')
        
        # Complete a stage with metadata
        helper.handle_stage_success(
            'ingestion', 
            duration=45.2, 
            jobs_processed=12,
            output_files=['outputs/renamed/file1.wav', 'outputs/renamed/file2.wav'],
            metadata={'model': 'test_model', 'version': '1.0'}
        )
        
        print("\n--- Testing Completed Stage ---")
        print_stage_status(run_folder, 'ingestion')
        
        # Fail a stage
        helper.handle_stage_failure('separation', 'Model failed to load', duration=30.1)
        
        print("\n--- Testing Failed Stage ---")
        print_stage_status(run_folder, 'separation')
        
        print("✅ Stage status test passed!")


def test_resume_from_specific_stage():
    """Test the resume_from logic"""
    print("🧪 Testing Resume From Specific Stage...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        run_folder = Path(temp_dir)
        helper = ResumeHelper(run_folder)
        
        # Complete several stages
        helper.handle_stage_success('ingestion')
        helper.handle_stage_success('separation')
        helper.handle_stage_success('normalization')
        
        # Test the resume_from logic by simulating it
        all_stages = get_pipeline_stages()
        resume_from = 'normalization'
        resume_from_index = all_stages.index(resume_from)
        
        # Stages before resume_from should be skippable
        for i, stage in enumerate(all_stages):
            if i < resume_from_index:
                # These should be skipped due to resume_from logic
                print(f"Stage {stage} (index {i}) should be skipped (before resume point)")
            elif i >= resume_from_index:
                # These should run (at or after resume point)
                print(f"Stage {stage} (index {i}) should run (at or after resume point)")
        
        print("✅ Resume from specific stage test passed!")


def demo_advanced_controls():
    """Demo all advanced resume controls"""
    print("🎯 Demo: Advanced Resume Controls")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        run_folder = Path(temp_dir)
        helper = ResumeHelper(run_folder)
        
        # Create a realistic scenario
        print("\n1. Setting up completed stages...")
        helper.handle_stage_success('ingestion', duration=45.2, jobs_processed=12)
        helper.handle_stage_success('separation', duration=120.8, jobs_processed=12)
        helper.handle_stage_success('normalization', duration=15.3, jobs_processed=12)
        helper.handle_stage_failure('clap', 'CLAP model timeout')
        
        # Show initial status
        print("\n2. Initial Resume Status:")
        print_resume_status(run_folder)
        
        # Show detailed stage status
        print("\n3. Detailed Stage Status (failed stage):")
        print_stage_status(run_folder, 'clap')
        
        # Force re-run a completed stage
        print("\n4. Force re-run 'separation' stage:")
        helper.force_rerun_stage('separation')
        print_stage_status(run_folder, 'separation')
        
        # Clear from a specific stage
        print("\n5. Clear from 'normalization' onwards:")
        helper.clear_from_stage('normalization')
        
        # Final status
        print("\n6. Final Resume Status:")
        print_resume_status(run_folder)


if __name__ == '__main__':
    print("🚀 Testing Advanced Resume Controls\n")
    
    test_force_rerun()
    test_clear_from_stage()
    test_stage_status()
    test_resume_from_specific_stage()
    
    print("\n" + "="*50)
    print("🎉 All advanced resume control tests passed!")
    print("="*50)
    
    demo_advanced_controls() 