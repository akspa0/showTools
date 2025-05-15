import pytest
from pathlib import Path
import sys
import os
from unittest import mock

# Ensure the utils directory is in the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import preset_utils
from utils import file_utils # Need sanitize_filename for comparison

# --- Fixture for mocking settings ---

@pytest.fixture
def mock_settings(tmp_path, monkeypatch):
    """Fixture to mock settings.CLAP_PRESETS_DIR to use tmp_path."""
    mock_presets_dir = tmp_path / "mock_presets"
    mock_presets_dir.mkdir()
    # Mock the setting within the preset_utils module where it's used
    monkeypatch.setattr(preset_utils.settings, "CLAP_PRESETS_DIR", mock_presets_dir)
    # Also mock the dependency from file_utils if needed within preset_utils
    monkeypatch.setattr(preset_utils, "settings", preset_utils.settings) 
    return mock_presets_dir

# --- Tests for load_clap_prompt_presets ---

def test_load_presets_empty_dir(mock_settings):
    """Test loading from an empty directory."""
    loaded = preset_utils.load_clap_prompt_presets()
    assert loaded == {}

def test_load_presets_valid_files(mock_settings):
    """Test loading valid preset files."""
    presets_dir = mock_settings
    # Create preset files
    (presets_dir / "animals.txt").write_text("dog\n cat \n bird\n")
    (presets_dir / "vehicles.txt").write_text("car\n  \ntruck\n") # Includes empty line
    (presets_dir / "_special_chars_.txt").write_text("test1\ntest2")
    (presets_dir / "other_ext.json").write_text("ignored") # Should be ignored
    
    loaded = preset_utils.load_clap_prompt_presets()
    
    assert len(loaded) == 3
    assert "animals" in loaded
    assert loaded["animals"] == ["dog", "cat", "bird"]
    assert "vehicles" in loaded
    assert loaded["vehicles"] == ["car", "truck"]
    assert "_special_chars_" in loaded # Check stem name handling
    assert loaded["_special_chars_"] == ["test1", "test2"]
    assert "other_ext" not in loaded

def test_load_presets_empty_file(mock_settings):
    """Test loading an empty or whitespace-only preset file."""
    presets_dir = mock_settings
    (presets_dir / "empty.txt").touch()
    (presets_dir / "whitespace.txt").write_text("  \n \n ")
    (presets_dir / "good.txt").write_text("hello")
    
    loaded = preset_utils.load_clap_prompt_presets()
    assert len(loaded) == 1
    assert "good" in loaded
    assert "empty" not in loaded
    assert "whitespace" not in loaded

# --- Tests for save_clap_prompt_preset ---

def test_save_preset_new(mock_settings):
    """Test saving a new preset file."""
    presets_dir = mock_settings
    preset_name = "My Test Preset"
    prompts = ["prompt 1", "  prompt 2  ", "prompt\n3"]
    expected_filename = "My Test Preset.txt" # Sanitization doesn't change this
    expected_path = presets_dir / expected_filename
    expected_content = "prompt 1\nprompt 2\nprompt\n3\n" # Note: strips surrounding whitespace, handles newline within prompt?
    # Correction: prompts should likely not contain literal newlines from UI
    prompts_corrected = ["prompt 1", "  prompt 2  ", "prompt 3"]
    expected_content_corrected = "prompt 1\nprompt 2\nprompt 3\n"

    assert not expected_path.exists()
    result = preset_utils.save_clap_prompt_preset(preset_name, prompts_corrected)
    assert result is True
    assert expected_path.exists()
    assert expected_path.read_text(encoding='utf-8') == expected_content_corrected

def test_save_preset_overwrite(mock_settings):
    """Test overwriting an existing preset file."""
    presets_dir = mock_settings
    preset_name = "Overwrite Me"
    initial_prompts = ["initial"]
    new_prompts = ["new prompt 1", "new prompt 2"]
    expected_filename = "Overwrite Me.txt"
    expected_path = presets_dir / expected_filename
    
    # Save initial version
    result1 = preset_utils.save_clap_prompt_preset(preset_name, initial_prompts)
    assert result1 is True
    assert expected_path.read_text(encoding='utf-8') == "initial\n"
    
    # Save new version
    result2 = preset_utils.save_clap_prompt_preset(preset_name, new_prompts)
    assert result2 is True
    assert expected_path.read_text(encoding='utf-8') == "new prompt 1\nnew prompt 2\n"

def test_save_preset_sanitization(mock_settings):
    """Test filename sanitization when saving."""
    presets_dir = mock_settings
    preset_name = "invalid:name?*.txt"
    prompts = ["test"]
    # Use the actual sanitize function for expected name
    sanitized_stem = file_utils.sanitize_filename(preset_name)
    expected_filename = f"{sanitized_stem}.txt" # e.g., "invalid_name_txt.txt"
    expected_path = presets_dir / expected_filename
    
    result = preset_utils.save_clap_prompt_preset(preset_name, prompts)
    assert result is True
    assert expected_path.exists()
    assert expected_path.name == expected_filename
    assert expected_path.read_text(encoding='utf-8') == "test\n"

def test_save_preset_empty_prompts(mock_settings):
    """Test saving with an empty list of prompts."""
    presets_dir = mock_settings
    preset_name = "Empty"
    prompts = []
    result = preset_utils.save_clap_prompt_preset(preset_name, prompts)
    assert result is False
    assert not list(presets_dir.glob("*.txt")) # Ensure no file was created

def test_save_preset_empty_name(mock_settings):
    """Test saving with an empty or whitespace-only preset name."""
    presets_dir = mock_settings
    prompts = ["test"]
    result_empty = preset_utils.save_clap_prompt_preset("", prompts)
    result_space = preset_utils.save_clap_prompt_preset("   ", prompts)
    assert result_empty is False
    assert result_space is False
    assert not list(presets_dir.glob("*.txt")) 