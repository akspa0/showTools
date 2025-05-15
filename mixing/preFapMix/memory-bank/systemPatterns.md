# System Patterns

## Architecture Overview
PreFapMix follows a streamlined pipeline architecture with these main components:
1. **Input Processing**: Reading, validating, and matching audio file pairs
2. **Stem Separation**: Separating vocals from instrumentals using python-audio-separator
3. **Audio Normalization**: Multi-stage loudness normalization.
    - Vocal and Instrumental stems normalized to target LUFS using `audiomentations`.
    - Optional peak limiting on vocal stems using `audiomentations`.
    - Final mixed output normalized to a target LUFS using `ffmpeg loudnorm`.
4. **Channel Separation**: Handling left/right channels
5. **Resampling**: Converting 8kHz audio to 44.1kHz as needed
6. **Transcription**: Using OpenAI Whisper for improved speech recognition
7. **Stereo Mixing**: Combining channels with proper positioning and adjustable instrumental volume
8. **Show Generation**: Creating consolidated show files with all calls in sequence
9. **Metadata Generation**: Creating detailed timestamps and call information
10. **Output Organization**: Saving files in chronologically structured directories

## Key Technical Decisions
- Moving from pydub to FFmpeg for core audio processing tasks (like final mix normalization, summing).
- Utilizing `audiomentations` for precise LUFS normalization and limiting of individual audio stems.
- Replacing fish-audio-processor with OpenAI Whisper for transcription
- Integrating python-audio-separator for stem separation
- Using FFmpeg's built-in resampling for better quality
- Applying loudness normalization through FFmpeg filters
- Using chronological organization for outputs
- Creating show files with detailed metadata for session-level analysis
- Using direct file matching for recv_out/trans_out pairs
- Prioritizing simplicity and functionality over complexity

## Design Patterns
- **Pipeline Processing**: Sequential processing of audio files
- **Command Pattern**: Using FFmpeg commands for audio manipulation
- **Factory Pattern**: Creating appropriate processing steps based on options
- **Strategy Pattern**: Different strategies for normalization and mixing
- **Observer Pattern**: Monitoring processing status and updating the UI accordingly
- **Facade Pattern**: Providing a simplified interface to complex subsystems

## Component Relationships
- **preFapMix.py**: Core processing logic and command-line interface
- **app.py**: Gradio-based web interface
- **file_matcher.py**: Logic for matching and organizing audio file pairs
- **stem_separator.py**: Interface to python-audio-separator
- **audio_processor.py**: Audio normalization, resampling, and mixing
- **transcriber.py**: Interface to OpenAI Whisper
- **show_builder.py**: Logic for creating consolidated show files with metadata
- **External Dependencies**:
  - FFmpeg: For audio processing
  - python-audio-separator: For stem separation
  - Whisper: For transcription 