# Active Context for PreFapMix

## Current Overall Goal

Successfully implement the `transcription_module.py` to parse RTTM diarization files correctly and generate detailed, `whisperBite.py`-style soundbite outputs (`.wav` and `.txt` files per speaker segment, organized in speaker-specific subfolders with PII-safe, content-derived names). This is the primary blocker for the entire audio processing pipeline.

## Current Focus & Immediate Task

**Resolving `AttributeError: type object 'Annotation' has no attribute 'read_rttm'` in `transcription_module.py`.**

*   **Problem:** The current attempt to load RTTM files using `pyannote.core.Annotation.read_rttm(diarization_file_path)` is failing because `read_rttm` is not a static method of the `Annotation` class in the way it's being called.
*   **Previous Attempt:** Using `pyannote.database.util.RTTMParser` also failed due to an `ImportError`, suggesting it might be from an incompatible version or deprecated.
*   **Current Strategy:** Analyze the attached `whisperBite.py` code (from `/c:/Users/akspa/showTools/mixing/preFapMix/v3.8/WhisperBite`) to understand how it successfully parses RTTM files and adapt `transcription_module.py` accordingly.

## Recent Changes & Discoveries

*   Identified that `Annotation.read_rttm()` as a static method call is incorrect.
*   Confirmed through logs that the transcription stage fails at RTTM loading, preventing soundbite generation and subsequent LLM summarization.
*   Updated `techContext.md` to reflect the RTTM parsing issue.
*   User has provided the `whisperBite.py` code for reference.

## Next Steps (Post-RTTM Fix)

1.  **Verify Soundbite Generation:** Once RTTM loading is fixed, ensure `transcription_module.py` correctly:
    *   Slices audio based on diarization.
    *   Transcribes each slice.
    *   Saves individual `.wav` and `.txt` soundbites to speaker-specific subfolders (e.g., `S0/`, `S1/`).
    *   Names soundbites using the `sequenceID_first_few_words` convention.
    *   Includes paths to these soundbites in the main transcript JSON.
2.  **Test `workflow_executor.py` End-to-End:** Run the full workflow with a sample audio file to ensure all stages complete successfully and outputs are generated as expected.
3.  **Test `call_processor.py`:**
    *   Provide output run directories from `workflow_executor.py` (for a paired call) to `call_processor.py`.
    *   Verify that `call_processor.py` correctly mixes audio, merges transcripts (preserving soundbite paths), and generates a combined LLM summary.
    *   Ensure `call_processor.py` correctly copies or links the soundbite directory structures from the individual stream processing outputs.
4.  **Address LM Studio Connectivity:** The logs showed an `httpx.ConnectError` for LM Studio. While secondary to the transcription issue, this will need to be resolved for LLM summarization to work (ensure LM Studio is running and accessible).

## Active Decisions & Considerations

*   **`pyannote.audio` Versioning/API:** The core of the current problem lies in correctly interfacing with the `pyannote.audio` library for RTTM parsing. The API or helper utilities for this task might have changed between versions or require a specific instantiation pattern not currently used. `whisperBite.py` is key to understanding the correct usage in the user's environment.
*   **Soundbite Naming & Structure:** Strict adherence to the `whisperBite.py` output format for soundbites is a key requirement.
*   **Error Propagation:** Ensure that if transcription fails, the workflow summary clearly indicates this and subsequent stages that depend on the transcript (like LLM summarization) are handled gracefully (e.g., skipped with a warning).

## Open Questions (for later)

*   Are there specific version requirements for `pyannote.audio` that `whisperBite.py` adheres to?
*   How should the system handle cases where diarization produces no segments or speaker labels?