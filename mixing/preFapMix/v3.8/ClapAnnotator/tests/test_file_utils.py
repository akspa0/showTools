import pytest
from pathlib import Path
from datetime import datetime
import re
import sys
import os

# Ensure the utils directory is in the Python path
# This might be necessary depending on how pytest discovers tests and the project structure
# Alternatively, run pytest with `python -m pytest` from the project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import the module to test
from utils import file_utils

# --- Tests for sanitize_filename ---

@pytest.mark.parametrize(
    "original, expected",
    [
        ("simple_filename.txt", "simple_filename_txt"),
        ("  leading and trailing spaces  ", "leading and trailing spaces"),
        ("file/with/slashes", "file_with_slashes"),
        ("file\\with\\backslashes", "file_with_backslashes"),
        ('file:with?"<>|*chars', "file_with_chars"),
        ("dot.at.end.", "dot_at_end_"),
        ("multiple__underscores__-", "multiple_underscores_-"), # Assuming dash is allowed
        ("", "invalid_filename"),
        ("   ", "invalid_filename"),
        (".", "_"), # Special case: just a dot
        ("..", "__") # Special case: double dot
    ]
)
def test_sanitize_filename(original, expected):
    assert file_utils.sanitize_filename(original) == expected

# --- Tests for ensure_dir ---

def test_ensure_dir(tmp_path):
    """Test creating a directory and ensuring it exists."""
    new_dir = tmp_path / "test_dir"
    assert not new_dir.exists()
    file_utils.ensure_dir(new_dir)
    assert new_dir.exists()
    assert new_dir.is_dir()
    # Test calling it again on existing dir (should not raise error)
    file_utils.ensure_dir(new_dir)
    assert new_dir.exists()

def test_ensure_dir_nested(tmp_path):
    """Test creating nested directories."""
    nested_dir = tmp_path / "parent" / "child"
    assert not nested_dir.exists()
    assert not (tmp_path / "parent").exists()
    file_utils.ensure_dir(nested_dir)
    assert nested_dir.exists()
    assert nested_dir.is_dir()
    assert (tmp_path / "parent").exists()

# Mock datetime for generate_output_path tests
class MockDateTime:
    @classmethod
    def now(cls):
        # Return a fixed datetime for predictable timestamps
        return datetime(2023, 10, 27, 10, 30, 0)

# --- Tests for generate_output_path ---

def test_generate_output_path(tmp_path, monkeypatch):
    """Test generating the timestamped output path."""
    # Mock datetime.now to return a fixed time
    monkeypatch.setattr("utils.file_utils.datetime", MockDateTime)
    
    base_output = tmp_path / "outputs"
    input_filename = "my_audio file.wav"
    
    expected_sanitized_stem = "my_audio file"
    expected_timestamp = "20231027_103000"
    expected_dir_name = f"{expected_sanitized_stem}_{expected_timestamp}"
    expected_path = base_output / expected_dir_name

    # Ensure base directory doesn't exist initially (ensure_dir inside will create it)
    assert not base_output.exists()
    
    generated_path = file_utils.generate_output_path(base_output, input_filename)
    
    assert generated_path == expected_path
    assert generated_path.exists()
    assert generated_path.is_dir()
    assert base_output.exists() # Check base directory was also created

def test_generate_output_path_sanitization(tmp_path, monkeypatch):
    """Test that filename sanitization is applied."""
    monkeypatch.setattr("utils.file_utils.datetime", MockDateTime)
    
    base_output = tmp_path / "outputs"
    input_filename = "weird/name?.wav"
    
    expected_sanitized_stem = "weird_name_"
    expected_timestamp = "20231027_103000"
    expected_dir_name = f"{expected_sanitized_stem}_{expected_timestamp}"
    expected_path = base_output / expected_dir_name
    
    generated_path = file_utils.generate_output_path(base_output, input_filename)
    assert generated_path == expected_path
    assert generated_path.exists()

# --- Tests for cleanup_directory ---

def test_cleanup_directory_exists(tmp_path):
    """Test cleaning up an existing directory with content."""
    dir_to_clean = tmp_path / "cleanup_me"
    dir_to_clean.mkdir()
    (dir_to_clean / "some_file.txt").touch()
    (dir_to_clean / "subdir").mkdir()
    (dir_to_clean / "subdir" / "another.dat").touch()
    
    assert dir_to_clean.exists()
    assert (dir_to_clean / "subdir").exists()
    
    file_utils.cleanup_directory(dir_to_clean)
    
    assert not dir_to_clean.exists()

def test_cleanup_directory_not_exists(tmp_path):
    """Test cleaning up a non-existent directory (should not error)."""
    dir_to_clean = tmp_path / "does_not_exist"
    assert not dir_to_clean.exists()
    
    # Should execute without raising an error
    file_utils.cleanup_directory(dir_to_clean)
    assert not dir_to_clean.exists()

def test_cleanup_directory_is_file(tmp_path):
    """Test attempting to clean up a file path (should not error)."""
    file_to_clean = tmp_path / "a_file.txt"
    file_to_clean.touch()
    assert file_to_clean.exists()
    assert file_to_clean.is_file()
    
    # Should execute without raising an error and leave file intact
    file_utils.cleanup_directory(file_to_clean)
    assert file_to_clean.exists() 