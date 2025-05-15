# PreFapMix Project Brief

## Core Requirements
- Process audio files (8kHz per channel) for transcription and analysis
- Match recv_out/trans_out file pairs that belong to the same conversation
- Properly resample each audio channel to 44.1kHz
- Separate vocals from instrumentals for clarity and better transcription
- Mix audio channels into proper stereo output preserving the original quality
- Allow adjustable instrumental volume in the final mix
- Transcribe vocal content using OpenAI Whisper (large-v3 model)
- Generate complete conversation transcripts with proper speaker attribution
- Create a chronologically ordered "show" file containing all processed calls
- Generate metadata with timestamps for each call in the show file

## Goals
- Fix existing sample rate conversion issues (currently mixing 8kHz channels then resampling the mix)
- Use ffmpeg for proper audio processing and normalization (including final mix loudness)
- Integrate python-audio-separator for vocal/instrumental separation
- Utilize audiomentations for effective stem loudness normalization
- Achieve a well-balanced mix with prominent vocals and controlled instrumental levels
- Ensure the final mixed output has appropriate and consistent loudness
- Replace fish-audio-processor (fap) with OpenAI Whisper for better transcription
- Generate chronologically ordered outputs
- Create show files that combine all calls in sequence with proper metadata
- Use transcription content for intelligent call naming
- Allow customization of instrumental volume in final mixes
- Create a simple, focused pipeline that borrows the best elements from previous tools
- Maintain a straightforward, efficient workflow without unnecessary complexity
- Provide a simple and efficient user interface for essential options
- Support batch processing of multiple audio files 