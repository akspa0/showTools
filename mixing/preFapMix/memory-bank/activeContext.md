# Active Context for PreFapMix

## Current Overall Goal

Successfully implement all stages of the audio processing workflow, including robust transcription, soundbite generation, and meaningful LLM-based summarization/analysis, ensuring outputs are accurate and meet user needs.

## Current Focus & Immediate Task

1.  **Diagnose and Resolve Terminal Echo Issue:** The terminal loses character echoing after `workflow_executor.py` runs, requiring a manual `reset`. This is the primary usability blocker.
    *   **Status:** Imports-only test (`test_imports.py`) does *not* cause the issue, indicating the problem lies in the execution of one or more stages within the workflow.
    *   **Strategy:** Systematically run `workflow_executor.py` with individual stages enabled/disabled to pinpoint the problematic module or library call.
2.  **Test and Verify LLM Summarization Functionality:** Now that `llm_module.py` imports correctly using `import lmstudio as lms` and catches specific `lms.LMStudioError` exceptions.
    *   **Strategy:** After resolving the terminal echo, run the full workflow and verify that the LLM stage connects to LM Studio, processes the transcript, and generates the expected summary output file.
3.  **Refine LLM Purpose and Prompts:** Based on user feedback and the initial definitions in `projectbrief.md` and `productContext.md`.

## Recent Changes & Discoveries

*   **LM Studio Integration Resolved (Imports):** `llm_module.py` now correctly imports `lmstudio as lms` and references its specific exception classes (e.g., `lms.LMStudioModelNotFoundError`, `lms.LMStudioServerError`) directly as attributes of the `lms` object. This was confirmed by successful import via `test_imports.py` after direct introspection of the `lms` module.
*   The `master_transcript.txt` generation is working correctly.
*   Terminal echo issue is confirmed *not* to occur from module imports alone.

## Next Steps

1.  **Isolate Terminal Echo Source:** Modify `workflow_executor.py` to test stages individually.
2.  **Fix Terminal Echo Issue.**
3.  **Conduct Full Workflow Test:** Verify all stages, including LLM summarization output.
4.  **Review and Refine LLM Prompts and `projectbrief.md`/`productContext.md`** with user input regarding the LLM's specific tasks and desired outputs.

## Active Decisions & Considerations

*   **Terminal Echo Culprit:** Suspects include `clap_module` (due to `audio-separator` external process), `audio_preprocessor` (`ffmpeg`), or other libraries that might manipulate terminal state without proper restoration.
*   **Defining LLM Utility (USER INPUT STILL KEY):** While imports are fixed, the *purpose* of the LLM summaries needs user validation and refinement to ensure the prompts and outputs are valuable.

## Open Questions

*   What are the precise, desired outputs from the LLM summarization stage? (e.g., bullet points, narrative summary, Q&A, entity extraction, sentiment analysis?)
*   Who is the end-user of these summaries, and what decisions will they inform?
*   Are there different types of summaries needed (e.g., per-stream vs. combined call)?
*   How should the system handle cases where diarization produces no segments or speaker labels (relevant for transcript quality fed to LLM)?