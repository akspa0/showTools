# Active Context for PreFapMix

## Current Overall Goal

Finalize all core audio processing and output generation features, including robust LLM-based naming, MP3 compression for individual calls, and implement the `show_compiler.py` script for creating aggregated "show files" from processed paired calls. Ensure robust PII safety and deliver user-friendly, organized outputs as defined in `projectbrief.md` and `productContext.md`.

## Current Focus & Immediate Tasks

1.  **LLM Model Update:**
    *   The LLM model for all summarization is now set to `llama-3.1-8b-supernova-etherealhermes` in the workflow config.
    *   All LLM outputs (call name, synopsis, hashtags) are always generated for every call/audio, with clear error reporting if LLM fails.
2.  **Final Output Naming:**
    *   The LLM-generated call name is now robustly used for the final output directory (quotes stripped, sanitized, fallback to call_id with clear logging).
    *   README and CLI now document `--input_file` for single-file workflows, and `--input_dir` for batch.
3.  **MP3 Compression:**
    *   MP3 compression for main audio is implemented; decision on soundbites pending.
4.  **Show Compiler:**
    *   Next focus: verify end-to-end output naming, then implement and test `show_compiler.py` for show-level aggregation.

## Recent Changes & Discoveries

*   **LLM Model and Output Naming:**
    *   Model is now `llama-3.1-8b-supernova-etherealhermes`.
    *   Final output directory naming is robust to LLM output quirks (quotes, empty, etc.).
    *   Logging improved for debugging naming issues.
*   **Documentation:**
    *   README and CLI usage updated for clarity and modern workflow.

## Next Steps (Sequential)

1.  Verify that the LLM-generated name is always used for the final output directory when possible.
2.  Implement and test `show_compiler.py` for show-level audio and transcript aggregation.
3.  Decide and implement if soundbites in the "final output" directories also require MP3 compression.
4.  Conduct thorough end-to-end testing of the complete workflow from `workflow_executor.py` through `call_processor.py` to `show_compiler.py`.
5.  Update all memory bank files to reflect these final additions and the overall completed state of this feature set.

## Active Decisions & Considerations

*   **Instrumental Stems for Show Audio:** Needs clarification – should `show_audio.mp3` include instrumental stems, or is the current vocal-only mix from `call_processor.py` sufficient? (Current assumption: vocal-only based on existing `mixed_vocals.wav`).
*   **MP3 Compression for Soundbites:** Decision pending on whether to compress the individual soundbite `.wav` files in the "final output" directory structure.

## Open Questions

*   Should soundbites in the `final_output_dir/[call_name]/soundbites/` also be MP3 compressed? (Assuming soundbites are planned to be copied there eventually).
*   Does the `show_audio.mp3` need to include instrumental stems, or is the current vocal-only mix from `call_processor.py` (i.e., `mixed_vocals.wav`) the correct source?