import pytest
from pathlib import Path
import sys
import os
from unittest import mock

# Mock the ffmpeg library before it's imported by audio_utils
# This is a common pattern when mocking external libraries
# Create mock objects for the functions/classes we expect to use
mock_ffmpeg_input = mock.MagicMock()
mock_ffmpeg_output = mock.MagicMock()
mock_ffmpeg_overwrite_output = mock.MagicMock()
mock_ffmpeg_run = mock.MagicMock()

# Configure the mock object chain input(...).output(...).overwrite_output(...).run(...)
mock_ffmpeg_input.return_value = mock_ffmpeg_output
mock_ffmpeg_output.output.return_value = mock_ffmpeg_overwrite_output
mock_ffmpeg_overwrite_output.overwrite_output.return_value = mock_ffmpeg_run

# Create a mock for the ffmpeg Error class
mock_ffmpeg_error = mock.Mock(spec=Exception) # Base exception is fine for type mocking

# Store the mock Error class where ffmpeg would normally expose it
mock_ffmpeg_module = mock.MagicMock(
    input=mock_ffmpeg_input,
    output=mock_ffmpeg_output,
    overwrite_output=mock_ffmpeg_overwrite_output,
    run=mock_ffmpeg_run,
    Error=mock_ffmpeg_error
)

# Use sys.modules to replace the actual ffmpeg module with our mock *before* audio_utils imports it
sys.modules['ffmpeg'] = mock_ffmpeg_module

# Ensure the utils directory is in the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import the module to test (it will import the mocked ffmpeg)
from utils import audio_utils

# --- Test Suite ---

@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset mocks before each test to prevent state leakage."""
    mock_ffmpeg_input.reset_mock()
    mock_ffmpeg_output.output.reset_mock()
    mock_ffmpeg_overwrite_output.overwrite_output.reset_mock()
    mock_ffmpeg_run.run.reset_mock()

def test_resample_audio_success(tmp_path):
    """Test successful resampling call."""
    input_file = tmp_path / "input.wav"
    output_file = tmp_path / "output.wav"
    target_sr = 16000
    
    # Create a dummy input file for the is_file() check
    input_file.touch()
    
    result_path = audio_utils.resample_audio_ffmpeg(input_file, output_file, target_sr)
    
    # Assert mocks were called correctly
    mock_ffmpeg_input.assert_called_once_with(str(input_file))
    mock_ffmpeg_output.output.assert_called_once_with(
        str(output_file),
        ar=str(target_sr),
        ac=1,
        acodec='pcm_s16le'
    )
    mock_ffmpeg_overwrite_output.overwrite_output.assert_called_once()
    mock_ffmpeg_run.run.assert_called_once_with(capture_stdout=True, capture_stderr=True, quiet=True)
    
    # Assert the output path is returned
    assert result_path == output_file

def test_resample_audio_input_not_found(tmp_path):
    """Test when the input file does not exist."""
    input_file = tmp_path / "non_existent.wav"
    output_file = tmp_path / "output.wav"
    target_sr = 16000
    
    with pytest.raises(FileNotFoundError):
        audio_utils.resample_audio_ffmpeg(input_file, output_file, target_sr)
    
    # Ensure ffmpeg was not called
    mock_ffmpeg_input.assert_not_called()

def test_resample_audio_ffmpeg_error(tmp_path, monkeypatch):
    """Test when the ffmpeg command itself raises an error."""
    input_file = tmp_path / "input.wav"
    output_file = tmp_path / "output.wav"
    target_sr = 16000
    input_file.touch()
    
    # Configure the mock run to raise the mocked ffmpeg.Error
    mock_error_instance = mock_ffmpeg_error("ffmpeg failed")
    # Attach a mock stderr attribute to the error instance
    mock_error_instance.stderr = b"Some detailed ffmpeg error"
    mock_ffmpeg_run.run.side_effect = mock_error_instance

    # Mock Path.exists and Path.unlink to test cleanup attempt
    mock_exists = mock.MagicMock(return_value=True) # Assume output file exists after failure
    mock_unlink = mock.MagicMock()
    monkeypatch.setattr(Path, "exists", mock_exists)
    monkeypatch.setattr(Path, "unlink", mock_unlink)
    
    with pytest.raises(RuntimeError, match="ffmpeg resampling failed: Some detailed ffmpeg error"):
        audio_utils.resample_audio_ffmpeg(input_file, output_file, target_sr)
        
    # Ensure ffmpeg run was called
    mock_ffmpeg_run.run.assert_called_once()
    
    # Ensure cleanup was attempted
    # We need to check the call against the specific output_file Path object
    assert mock_exists.call_args_list[0][0][0] == output_file 
    assert mock_unlink.call_args_list[0][0][0] == output_file

def test_resample_audio_ffmpeg_error_no_stderr(tmp_path):
    """Test ffmpeg error when stderr is None."""
    input_file = tmp_path / "input.wav"
    output_file = tmp_path / "output.wav"
    target_sr = 16000
    input_file.touch()
    
    mock_error_instance = mock_ffmpeg_error("ffmpeg failed")
    mock_error_instance.stderr = None # Simulate no stderr
    mock_ffmpeg_run.run.side_effect = mock_error_instance
    
    with pytest.raises(RuntimeError, match="ffmpeg resampling failed: No stderr output"):
        audio_utils.resample_audio_ffmpeg(input_file, output_file, target_sr) 