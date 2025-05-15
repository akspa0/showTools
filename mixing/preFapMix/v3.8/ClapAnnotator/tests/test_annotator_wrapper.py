import pytest
from unittest.mock import patch, MagicMock, call
import numpy as np
import torch
import librosa

# Mock settings before importing the annotator
# We might need more sophisticated mocking later using fixtures/monkeypatch
class MockSettings:
    CLAP_MODEL_ID = "mock/clap-model"
    CLAP_EXPECTED_SR = 48000
    CLAP_CHUNK_DURATION_S = 10
    DEVICE = "cpu" # Use CPU for testing
    BASE_OUTPUT_DIR = "test_output"

@pytest.fixture(autouse=True)
def mock_settings_fixture(monkeypatch):
    # Use monkeypatch to replace the actual settings module during tests
    monkeypatch.setattr('config.settings', MockSettings)
    # Ensure the annotator module uses the mocked settings when imported
    import sys
    if 'clap_annotation.annotator' in sys.modules:
        import importlib
        importlib.reload(sys.modules['clap_annotation.annotator'])

# Import the class under test *after* potentially mocking settings
from clap_annotation.annotator import CLAPAnnotatorWrapper

# Fixture to create a CLAPAnnotatorWrapper instance with mocked dependencies
@pytest.fixture
def mock_annotator():
    with patch('clap_annotation.annotator.ClapModel') as MockClapModel, \\
         patch('clap_annotation.annotator.ClapProcessor') as MockClapProcessor, \\
         patch('clap_annotation.annotator.torch') as MockTorch, \\
         patch('clap_annotation.annotator.settings', MockSettings): # Ensure settings are mocked

        # Configure mock model and processor instances
        mock_model_instance = MagicMock()
        mock_processor_instance = MagicMock()
        MockClapModel.from_pretrained.return_value = mock_model_instance
        MockClapProcessor.from_pretrained.return_value = mock_processor_instance

        # Mock torch device functionality if needed (assuming basic check)
        MockTorch.device.return_value = MockSettings.DEVICE
        mock_model_instance.to.return_value = mock_model_instance # Simulate moving to device

        annotator = CLAPAnnotatorWrapper()
        # Attach mocks for inspection in tests
        annotator.mock_model = mock_model_instance
        annotator.mock_processor = mock_processor_instance
        annotator.mock_torch = MockTorch
        annotator.MockClapModel = MockClapModel
        annotator.MockClapProcessor = MockClapProcessor
        return annotator

# --- Test Cases ---

def test_annotator_initialization(mock_annotator):
    """Test if the annotator initializes correctly and loads model/processor."""
    mock_annotator.MockClapModel.from_pretrained.assert_called_once_with(MockSettings.CLAP_MODEL_ID)
    mock_annotator.MockClapProcessor.from_pretrained.assert_called_once_with(MockSettings.CLAP_MODEL_ID)
    mock_annotator.mock_model.to.assert_called_once_with(MockSettings.DEVICE)
    assert mock_annotator.model == mock_annotator.mock_model
    assert mock_annotator.processor == mock_annotator.mock_processor
    assert mock_annotator.device == MockSettings.DEVICE
    assert mock_annotator.expected_sr == MockSettings.CLAP_EXPECTED_SR

def test_get_text_features(mock_annotator):
    """Test the _get_text_features method."""
    prompts = ["prompt one", "prompt two"]
    mock_processed_text = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
    mock_text_features = MagicMock(spec=torch.Tensor)

    mock_annotator.mock_processor.return_value = mock_processed_text
    mock_annotator.mock_model.get_text_features.return_value = mock_text_features

    # Mock tensor moving to device
    mock_processed_text["input_ids"].to.return_value = mock_processed_text["input_ids"]
    mock_processed_text["attention_mask"].to.return_value = mock_processed_text["attention_mask"]

    features = mock_annotator._get_text_features(prompts)

    mock_annotator.mock_processor.assert_called_once_with(
        text=prompts, padding=True, truncation=True, return_tensors="pt"
    )
    # Check that inputs were moved to the correct device
    mock_processed_text["input_ids"].to.assert_called_once_with(mock_annotator.device)
    mock_processed_text["attention_mask"].to.assert_called_once_with(mock_annotator.device)

    # Check that model's method was called with the device-specific tensors
    mock_annotator.mock_model.get_text_features.assert_called_once_with(
        input_ids=mock_processed_text["input_ids"],
        attention_mask=mock_processed_text["attention_mask"]
    )
    assert features is mock_text_features

def test_process_audio_chunk(mock_annotator):
    """Test the _process_audio_chunk method."""
    # Create realistic mock audio data (e.g., 1 second at 48kHz)
    mock_audio_chunk = np.random.randn(MockSettings.CLAP_EXPECTED_SR).astype(np.float32)
    mock_prompts = ["prompt 1", "prompt 2"]
    # Mock text features (shape: [batch_size=1, num_prompts=2, feature_dim=512])
    # Needs realistic shape for dot product
    mock_text_features = torch.randn(1, len(mock_prompts), 512)

    mock_processed_audio = {"input_features": MagicMock()}
    # Mock audio features (shape: [batch_size=1, 1, feature_dim=512]) - Needs realistic shape
    mock_audio_features = torch.randn(1, 1, 512)
    # Mock logits (shape: [batch_size=1, num_prompts=2]) - result of matmul
    mock_logits = torch.tensor([[0.1, 2.5]]) # Example logits before softmax/sigmoid
    # Mock probabilities after applying activation (e.g., sigmoid)
    mock_probs = torch.sigmoid(mock_logits) # tensor([[0.5249, 0.9241]])

    mock_annotator.mock_processor.return_value = mock_processed_audio
    mock_annotator.mock_model.get_audio_features.return_value = mock_audio_features
    mock_annotator.mock_torch.no_grad.return_value.__enter__.return_value = None # Context manager
    mock_annotator.mock_torch.no_grad.return_value.__exit__.return_value = None

    # Mock the matrix multiplication (dot product)
    # We need to carefully mock the sequence: audio_features @ text_features.T
    # Result shape should be (batch=1, n_prompts=2)
    mock_audio_features.matmul.return_value.squeeze.return_value = mock_logits

    # Mock the activation function (e.g., sigmoid)
    mock_annotator.mock_torch.sigmoid.return_value = mock_probs

    results = mock_annotator._process_audio_chunk(mock_audio_chunk, mock_prompts, mock_text_features)

    mock_annotator.mock_processor.assert_called_once_with(
        audios=mock_audio_chunk, sampling_rate=mock_annotator.expected_sr, return_tensors="pt"
    )
    # Check tensor moved to device
    mock_processed_audio["input_features"].to.assert_called_once_with(mock_annotator.device)

    mock_annotator.mock_model.get_audio_features.assert_called_once_with(
        mock_processed_audio["input_features"]
    )

    # Verify similarity calculation steps
    mock_audio_features.matmul.assert_called_once() # Check matmul was called
    # Check the operand for matmul was the transpose of text_features
    assert mock_audio_features.matmul.call_args[0][0] is mock_text_features.T
    mock_annotator.mock_torch.sigmoid.assert_called_once_with(mock_logits) # Check activation

    expected_scores = mock_probs[0].cpu().numpy() # Get numpy array from the mocked torch tensor
    assert np.allclose(results['scores'], expected_scores)
    assert results['labels'] == mock_prompts

# Remove placeholder test if it exists
# def test_placeholder():
#    """ Placeholder test to ensure the file is created correctly. """
#    assert True

# Add test for the main annotate method next

@patch('clap_annotation.annotator.librosa.load')
@patch.object(CLAPAnnotatorWrapper, '_get_text_features')
@patch.object(CLAPAnnotatorWrapper, '_process_audio_chunk')
def test_annotate_method(
    mock_process_chunk, mock_get_text_features, mock_librosa_load, mock_annotator
):
    """Test the main annotate method with chunking and filtering."""
    audio_file_path = "dummy/audio.wav"
    prompts = ["speech", "music"]
    threshold = 0.7
    mock_progress_callback = MagicMock()

    # --- Mock Data Setup ---
    # Simulate 25 seconds of audio at the expected SR (2.5 chunks)
    total_duration_s = 25
    mock_sr = MockSettings.CLAP_EXPECTED_SR
    mock_audio_data = np.random.randn(total_duration_s * mock_sr).astype(np.float32)
    chunk_duration_s = MockSettings.CLAP_CHUNK_DURATION_S
    samples_per_chunk = chunk_duration_s * mock_sr

    mock_librosa_load.return_value = (mock_audio_data, mock_sr)

    # Mock text features (doesn't need specific content for this test)
    mock_text_features = torch.randn(1, len(prompts), 512)
    mock_get_text_features.return_value = mock_text_features

    # Mock results from processing each chunk
    # Chunk 1 (0-10s): High confidence for speech
    mock_process_chunk.side_effect = [
        {
            'scores': np.array([0.9, 0.2]), # speech, music
            'labels': prompts
        },
        {
            'scores': np.array([0.1, 0.8]), # speech, music
            'labels': prompts
        },
        {
            'scores': np.array([0.6, 0.5]), # speech, music (below threshold)
            'labels': prompts
        }
    ]

    # --- Execute --- 
    results = mock_annotator.annotate(
        audio_file_path,
        prompts,
        threshold=threshold,
        progress_callback=mock_progress_callback
    )

    # --- Assertions ---
    # Check librosa.load call
    mock_librosa_load.assert_called_once_with(audio_file_path, sr=mock_sr, mono=True)

    # Check text feature extraction call
    mock_get_text_features.assert_called_once_with(prompts)

    # Check audio chunk processing calls
    assert mock_process_chunk.call_count == 3
    calls = mock_process_chunk.call_args_list
    # Check first chunk data and prompts
    np.testing.assert_array_almost_equal(calls[0][0][0], mock_audio_data[:samples_per_chunk])
    assert calls[0][0][1] == prompts
    assert calls[0][0][2] is mock_text_features
    # Check second chunk data
    np.testing.assert_array_almost_equal(calls[1][0][0], mock_audio_data[samples_per_chunk:2*samples_per_chunk])
    # Check third (partial) chunk data
    np.testing.assert_array_almost_equal(calls[2][0][0], mock_audio_data[2*samples_per_chunk:])

    # Check progress callback calls
    assert mock_progress_callback.call_count == 3
    mock_progress_callback.assert_has_calls([
        call(1, 3, 33.33), # chunk 1 / 3
        call(2, 3, 66.67), # chunk 2 / 3
        call(3, 3, 100.00) # chunk 3 / 3
    ], any_order=False)

    # Check final aggregated results (only detections above threshold)
    expected_results = [
        {
            "start_time": 0.0,
            "end_time": 10.0,
            "label": "speech",
            "score": 0.9
        },
        {
            "start_time": 10.0,
            "end_time": 20.0,
            "label": "music",
            "score": 0.8
        }
        # Chunk 3 results ([0.6, 0.5]) are below threshold 0.7
    ]

    # Use np.testing for score comparison due to potential float inaccuracies
    assert len(results) == len(expected_results)
    for i, res in enumerate(results):
        assert res["start_time"] == expected_results[i]["start_time"]
        assert res["end_time"] == expected_results[i]["end_time"]
        assert res["label"] == expected_results[i]["label"]
        np.testing.assert_almost_equal(res["score"], expected_results[i]["score"], decimal=5)

@patch('clap_annotation.annotator.librosa.load')
@patch.object(CLAPAnnotatorWrapper, '_get_text_features')
@patch.object(CLAPAnnotatorWrapper, '_process_audio_chunk')
def test_annotate_short_audio(
    mock_process_chunk, mock_get_text_features, mock_librosa_load, mock_annotator
):
    """Test the annotate method with audio shorter than one chunk."""
    audio_file_path = "dummy/short_audio.wav"
    prompts = ["clap"]
    threshold = 0.5
    mock_progress_callback = MagicMock()

    # Simulate 5 seconds of audio
    total_duration_s = 5
    mock_sr = MockSettings.CLAP_EXPECTED_SR
    mock_audio_data = np.random.randn(total_duration_s * mock_sr).astype(np.float32)

    mock_librosa_load.return_value = (mock_audio_data, mock_sr)
    mock_text_features = torch.randn(1, len(prompts), 512)
    mock_get_text_features.return_value = mock_text_features

    # Mock result for the single chunk
    mock_process_chunk.return_value = {
        'scores': np.array([0.85]), # clap
        'labels': prompts
    }

    results = mock_annotator.annotate(
        audio_file_path, prompts, threshold=threshold, progress_callback=mock_progress_callback
    )

    mock_librosa_load.assert_called_once_with(audio_file_path, sr=mock_sr, mono=True)
    mock_get_text_features.assert_called_once_with(prompts)
    # Should only process one chunk
    assert mock_process_chunk.call_count == 1
    np.testing.assert_array_almost_equal(mock_process_chunk.call_args[0][0], mock_audio_data)
    assert mock_process_chunk.call_args[0][1] == prompts
    assert mock_process_chunk.call_args[0][2] is mock_text_features

    # Progress callback should be called once
    mock_progress_callback.assert_called_once_with(1, 1, 100.0)

    # Check results
    expected_results = [
        {
            "start_time": 0.0,
            "end_time": 5.0, # Duration of the audio
            "label": "clap",
            "score": 0.85
        }
    ]
    assert len(results) == len(expected_results)
    assert results[0]["start_time"] == expected_results[0]["start_time"]
    assert results[0]["end_time"] == expected_results[0]["end_time"]
    assert results[0]["label"] == expected_results[0]["label"]
    np.testing.assert_almost_equal(results[0]["score"], expected_results[0]["score"], decimal=5)

@patch('clap_annotation.annotator.librosa.load')
@patch.object(CLAPAnnotatorWrapper, '_get_text_features')
@patch.object(CLAPAnnotatorWrapper, '_process_audio_chunk')
def test_annotate_no_results_above_threshold(
    mock_process_chunk, mock_get_text_features, mock_librosa_load, mock_annotator
):
    """Test annotate when no chunk results meet the threshold."""
    audio_file_path = "dummy/quiet_audio.wav"
    prompts = ["sound"]
    threshold = 0.9
    mock_progress_callback = MagicMock()

    # Simulate 12 seconds of audio (2 chunks)
    total_duration_s = 12
    mock_sr = MockSettings.CLAP_EXPECTED_SR
    mock_audio_data = np.random.randn(total_duration_s * mock_sr).astype(np.float32)

    mock_librosa_load.return_value = (mock_audio_data, mock_sr)
    mock_text_features = torch.randn(1, len(prompts), 512)
    mock_get_text_features.return_value = mock_text_features

    # Mock results below threshold
    mock_process_chunk.side_effect = [
        {'scores': np.array([0.8]), 'labels': prompts},
        {'scores': np.array([0.5]), 'labels': prompts}
    ]

    results = mock_annotator.annotate(
        audio_file_path, prompts, threshold=threshold, progress_callback=mock_progress_callback
    )

    assert mock_process_chunk.call_count == 2
    mock_progress_callback.assert_has_calls([
        call(1, 2, 50.0), call(2, 2, 100.0)
    ])
    # No results should be returned
    assert results == [] 