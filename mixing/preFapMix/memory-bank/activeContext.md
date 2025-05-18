# Active Context for PreFapMix

## Current Overall Goal

Finalize all core audio processing and output generation features, including robust, extensible LLM-based outputs, MP3 compression for individual calls, and implement the `show_compiler.py` script for creating aggregated "show files" from processed paired calls. Ensure robust PII safety and deliver user-friendly, organized outputs as defined in `projectbrief.md` and `productContext.md`.

## Major New Features (May 2025)

- **Stage 0 Metadata Harvesting:**
  - Every input file is analyzed for bitrate, channels, sample rate, codec, duration, and format using ffprobe or similar.
  - Metadata is saved as a JSON file alongside the input and propagated through the pipeline, updated at each stage.
- **Automatic Apollo Restoration:**
  - If an input is MP3 and (≤96kbps stereo or ≤64kbps mono), Apollo restoration is run before the main pipeline.
  - Restoration status is tracked in metadata and output tags.
- **Rich Output Tagging:**
  - All final outputs (MP3s, soundbites, etc.) include:
    - ID3 tags: title, categories (TXXX:Categories), genre (TCON), Apollo flag (TXXX:ApolloRestored), full lineage (TXXX:Lineage), and synopsis/comment (COMM).
    - Sidecar JSON: all metadata, including processing lineage and Apollo status.
  - LLM-generated categories/tags are embedded in ID3 for easy sorting/organization.
- **Soundbite Lineage:**
  - Each soundbite output includes source file, start/end timestamps, transcript, speaker label, and Apollo flag in tags/sidecar.
  - Speaker labels are flexible (not limited to S0–S7); log warning if >8 speakers.
- **Future-Proofing:**
  - System is ready for show-level LLM synopses and show compiler integration.
  - Documentation and README will be updated to reflect all new features and tagging conventions.
- **Extensible LLM Task System:**
  - The LLM stage in the workflow config now supports an arbitrary list of LLM tasks (prompt, output file, etc.), not just name/synopsis/categories.
  - Users can add new LLM-powered utilities (e.g., image prompts, songs, custom summaries) by editing the workflow JSON.
  - This system is designed for future UI exposure, empowering users to define new LLM tasks without code changes.
  - Backward compatibility: old configs with a single prompt are auto-converted internally.
- **CLAP Annotation Refactor:**
  - CLAP annotation will now always run on the original, unsegmented audio file (not on pre-processed or split audio).
  - Plan to refactor and integrate the CLAP annotation logic directly into the main project (as `clap_annotator.py`), removing the dependency on `v3.8/ClapAnnotator`.
  - The pipeline will use a single, well-structured CLAP annotation JSON for all downstream segmentation and analysis.

## Current Focus & Immediate Tasks

1.  **Final Output System Refactor (Top Priority):**
    *   The new final output builder system is now fully implemented and invoked unconditionally as the last step of the pipeline:
        - All calls are finalized (with optional tones appended to the end of each call audio).
        - Soundbites are converted to MP3 with ID3 and JSON metadata.
        - All LLM responses and transcripts are parsed and organized for each call.
        - A show file (MP3) is built by concatenating the finalized call audios, with a show transcript and timestamps.
        - Comprehensive metadata is written as a single JSON file, with all paths relative to the project root.
        - The builder always produces a robust, user-facing output folder with all required files, metadata, and a zipped archive.
        - All lineage, tagging, and extensibility requirements are met.
        - Content moderation (censoring or flagging problematic content), show-level LLM synopses, and UI integration for LLM tasks are the next focus.
    *   **NEW:** Output cleanup logic is now implemented in character_ai_description_builder.py: any call output folder under 192KB is automatically deleted after processing. This reduces clutter from empty or failed LLM runs. This logic is now also implemented in the final output builder.
    *   **NEW:** Output folders are now renamed using sanitized call names from call_title.txt or *_suggested_name.txt, matching the final output builder's naming logic (punctuation removed, 8-word limit, underscores, fallback to folder name, uniqueness enforced). This is now consistent across both character and final output builders.
2.  **LLM Model & Extensibility:**
    *   The LLM model for all summarization is now set to `llama-3.1-8b-supernova-etherealhermes` in the workflow config.
    *   All LLM outputs (call name, synopsis, categories, and any user-defined tasks) are generated for every call/audio, with clear error reporting if LLM fails.
    *   LLM tag output is now a comma-separated list of plain English categories (not hashtags), as per new prompt.
    *   The LLM task system is now fully extensible and user-driven.
3.  **Final Output Directory:**
    *   The final output is now always written to `05_final_output/` in the main output directory, no user config needed.
    *   The LLM-generated call name is robustly used for the final output directory (quotes stripped, sanitized, fallback to call_id with clear logging).
    *   README and CLI now document `--input_file` for single-file workflows, and `--input_dir` for batch.
4.  **MP3 Compression:**
    *   MP3 compression for main audio is implemented; decision on soundbites pending.
5.  **Show Compiler:**
    *   Next focus: test and validate the new LLM task system, then implement and test `show_compiler.py` for show-level aggregation.
6.  **CLAP-Based Call Segmentation (Planned):**
    *   Design and implement rules-based segmentation using the new CLAP annotation output, with rules defined in the workflow config.
    *   Refactor pipeline to use the new integrated `clap_annotator.py`.

## Recent Changes & Discoveries

*   **LLM Model and Output Naming:**
    *   Model is now `llama-3.1-8b-supernova-etherealhermes`.
    *   Final output directory naming is robust to LLM output quirks (quotes, empty, etc.).
    *   Logging improved for debugging naming issues.
*   **Output Cleanup & Naming:**
    *   Output folders under 192KB are now deleted after LLM processing in both character_ai_description_builder.py and final_output_builder.py. This helps keep results clean and removes empty/bad call folders.
    *   Output folders are now renamed using sanitized call names from call_title.txt or *_suggested_name.txt, matching the final output builder's naming logic (punctuation removed, 8-word limit, underscores, fallback to folder name, uniqueness enforced). This is now consistent across both character and final output builders.
*   **Documentation:**
    *   README and CLI usage updated for clarity and modern workflow.
    *   LLM extensibility and prompt templating now documented.
    *   CLAP annotation and segmentation refactor planned.

## Next Steps

- Refactor and integrate CLAP annotation logic into the main project.
- Implement rules-based segmentation using the new CLAP annotation output.
- Test and validate the new LLM task system with a real workflow run.
- Plan and begin UI integration for LLM task management.
- Propagate and update metadata through all pipeline stages.
- Implement and test ID3/JSON tagging for all outputs and soundbites.
- Update documentation and README.
- Begin work on show-level LLM synopses and show compiler integration.

## Active Decisions & Considerations

*   **Instrumental Stems for Show Audio:** Needs clarification – should `show_audio.mp3` include instrumental stems, or is the current vocal-only mix from `call_processor.py` sufficient? (Current assumption: vocal-only based on existing `mixed_vocals.wav`).
*   **MP3 Compression for Soundbites:** Decision pending on whether to compress the individual soundbite `.wav` files in the "final output" directory structure.

## Open Questions

*   Should soundbites in the `final_output_dir/[call_name]/soundbites/` also be MP3 compressed? (Assuming soundbites are planned to be copied there eventually).
*   Does the `show_audio.mp3` need to include instrumental stems, or is the current vocal-only mix from `call_processor.py` (i.e., `mixed_vocals.wav`) the correct source?

## Final Output Builder Redesign (May 2025)
- The final output builder is being completely redesigned.
- The new design will:
  - Recursively discover all processed call folders in the workflow output tree.
  - For each call:
    - Extract and sanitize the LLM-generated call title (remove punctuation, limit to 8 words, fallback to call_id), append timestamp.
    - Locate or build the final mixed audio (mix vocals + instrumental at 50% if needed), compress to MP3, tag with full metadata, and place in the root output folder.
    - Copy all transcript files to Transcriptions/<CallName>/.
    - Copy/convert all soundbites to MP3, organize by speaker in Soundbites/<CallName>/.
    - Copy all LLM outputs and metadata to Metadata/<CallName>/.
  - Write final_output_metadata.json and bad_calls.txt.
  - Zip the output folder.
  - Add robust logging and error handling.
- The output structure will be flat, user-facing, and future-proof, with all calls in the root and supporting folders for transcripts, soundbites, and metadata.
- This will be implemented in a new session.