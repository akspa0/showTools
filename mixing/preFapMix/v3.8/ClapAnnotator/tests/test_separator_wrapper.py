import pytest
from pathlib import Path
import sys
from unittest import mock

# Mock the audio_separator library before it's imported
mock_separator_instance = mock.MagicMock()
mock_separator_class = mock.MagicMock(return_value=mock_separator_instance)

sys.modules['audio_separator.separator'] = mock.MagicMock(
    Separator=mock_separator_class
)

# Mock dependencies used by the module under test
sys.modules['config'] = mock.MagicMock()
sys.modules['utils.file_utils'] = mock.MagicMock()

# Ensure the project root is in the Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import the module to test
from audio_separation import separator

# --- Test Suite ---

@pytest.fixture(autouse=True)
def reset_all_mocks():
    """Reset mocks before each test."""
    mock_separator_class.reset_mock()
    mock_separator_instance.reset_mock()
    # Reset mocks for dependencies if they were called
    sys.modules['config'].reset_mock()
    sys.modules['utils.file_utils'].ensure_dir.reset_mock()

@pytest.fixture
def mock_settings_paths(monkeypatch, tmp_path):
    """Fixture to provide mock paths via mocked settings."""
    mock_temp_dir = tmp_path / "temp_output"
    mock_model_dir = tmp_path / "models"
    # Mock the settings directly within the separator module's namespace
    monkeypatch.setattr(separator.settings, "TEMP_OUTPUT_DIR", mock_temp_dir)
    monkeypatch.setattr(separator.settings, "AUDIO_SEPARATOR_MODEL_DIR", mock_model_dir)
    return mock_temp_dir, mock_model_dir

def test_separator_wrapper_init_success(mock_settings_paths):
    """Test successful initialization of the wrapper."""
    mock_temp_dir, mock_model_dir = mock_settings_paths
    model_name = "UVR_MDXNET_Main"
    log_level = "DEBUG"
    extra_params = {"mdx_params": {"segment_size": 512}, "output_format": "FLAC"}
    
    wrapper = separator.AudioSeparatorWrapper(
        model_name=model_name,
        log_level=log_level,
        **extra_params
    )
    
    # Check ensure_dir was called
    separator.ensure_dir.assert_any_call(mock_temp_dir)
    separator.ensure_dir.assert_any_call(mock_model_dir)
    
    # Check Separator class was instantiated correctly
    expected_options = {
        "use_soundfile": True, # Default added by wrapper
        "sample_rate": 44100, # Default added by wrapper
        "mdx_params": {"segment_size": 512}, # From extra_params
        "output_format": "FLAC" # From extra_params
    }
    mock_separator_class.assert_called_once_with(
        model_name=model_name,
        model_file_dir=str(mock_model_dir),
        output_dir=str(mock_temp_dir),
        log_level=log_level,
        log_formatter=None,
        **expected_options
    )
    assert wrapper.separator == mock_separator_instance

def test_separator_wrapper_init_failure(mock_settings_paths):
    """Test initialization failure when Separator class errors."""
    mock_separator_class.side_effect = ValueError("Model init failed")
    
    with pytest.raises(RuntimeError, match="Failed to initialize Separator: Model init failed"):
        separator.AudioSeparatorWrapper(model_name="AnyModel")

@mock.patch.object(separator.Path, 'is_file')
def test_separator_separate_success(mock_is_file, mock_settings_paths):
    """Test successful separation call."""
    mock_temp_dir, _ = mock_settings_paths
    mock_is_file.return_value = True # Make input file appear to exist
    input_file = Path("path/to/input.wav")
    model_name = "CoolModel"
    
    wrapper = separator.AudioSeparatorWrapper(model_name=model_name)

    # Define expected output filenames based on logic in separate()
    expected_vocal_fname = f"{input_file.stem}_(Vocals)_{model_name}.wav"
    expected_instr_fname = f"{input_file.stem}_(Instrumental)_{model_name}.wav"
    expected_vocal_path = mock_temp_dir / expected_vocal_fname
    expected_instr_path = mock_temp_dir / expected_instr_fname
    
    # Mock the return value of the underlying separator's separate method
    mock_separator_instance.separate.return_value = [
        str(expected_vocal_path),
        str(expected_instr_path)
    ]
    
    result = wrapper.separate(input_file)
    
    # Check the underlying separate was called
    expected_output_names = {
        "Vocals": f"{input_file.stem}_(Vocals)_{model_name}",
        "Instrumental": f"{input_file.stem}_(Instrumental)_{model_name}"
    }
    mock_separator_instance.separate.assert_called_once_with(str(input_file), output_names=expected_output_names)
    
    # Check the returned dictionary
    assert result == {
        "Vocals": expected_vocal_path,
        "Instrumental": expected_instr_path
    }

@mock.patch.object(separator.Path, 'is_file')
def test_separator_separate_missing_stem(mock_is_file, mock_settings_paths):
    """Test separation when an expected stem is missing from output."""
    mock_temp_dir, _ = mock_settings_paths
    mock_is_file.return_value = True
    input_file = Path("input.flac")
    model_name = "OnlyVox"
    wrapper = separator.AudioSeparatorWrapper(model_name=model_name)
    
    expected_vocal_fname = f"{input_file.stem}_(Vocals)_{model_name}.flac"
    expected_vocal_path = mock_temp_dir / expected_vocal_fname
    
    # Simulate separator only returning the vocal stem
    mock_separator_instance.separate.return_value = [str(expected_vocal_path)]
    
    result = wrapper.separate(input_file)
    
    assert result == {"Vocals": expected_vocal_path}
    # Check logs would show a warning for missing Instrumental stem (requires log capture)

@mock.patch.object(separator.Path, 'is_file')
def test_separator_separate_no_matching_stems(mock_is_file, mock_settings_paths):
    """Test separation when NO expected stems are returned."""
    mock_is_file.return_value = True
    input_file = Path("input.m4a")
    model_name = "WeirdModel"
    wrapper = separator.AudioSeparatorWrapper(model_name=model_name)
    
    # Simulate separator returning unexpected files
    mock_separator_instance.separate.return_value = ["other_output.wav", "bass_output.wav"]
    
    result = wrapper.separate(input_file)
    
    assert result == {}
    # Check logs would show errors/warnings (requires log capture)

@mock.patch.object(separator.Path, 'is_file')
def test_separator_separate_input_not_found(mock_is_file, mock_settings_paths):
    """Test separation when input file does not exist."""
    mock_is_file.return_value = False # Simulate file not existing
    input_file = Path("non_existent.wav")
    wrapper = separator.AudioSeparatorWrapper(model_name="Any")
    
    with pytest.raises(FileNotFoundError):
        wrapper.separate(input_file)
    
    mock_separator_instance.separate.assert_not_called()

@mock.patch.object(separator.Path, 'is_file')
def test_separator_separate_runtime_error(mock_is_file, mock_settings_paths):
    """Test separation when the underlying separator call fails."""
    mock_is_file.return_value = True
    input_file = Path("input.wav")
    wrapper = separator.AudioSeparatorWrapper(model_name="Any")
    
    # Simulate error during separation
    mock_separator_instance.separate.side_effect = Exception("Separation crashed")
    
    with pytest.raises(RuntimeError, match="Audio separation failed: Separation crashed"):
        wrapper.separate(input_file) 