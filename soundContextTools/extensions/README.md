# Extension System for Audio Context Tool

## Overview
Extensions allow you to add post-processing, context generation, and derivative art/analytics to the finalized outputs of the pipeline. Extensions run after the main pipeline and operate only on anonymized, finalized data.

## Character Persona Builder Extension

- The `character_persona_builder.py` extension generates advanced Character.AI persona definitions for each call/channel/speaker using transcripts and LLMs.
- **Channel folders may have run-specific prefixes** (e.g., `0000-conversation`). The extension normalizes these for output and is robust to naming.
- **For conversation-only calls:** Generates a separate persona for each detected speaker (no merging).
- **For left/right calls:** Merges all speakers per channel and generates one persona per channel.
- **System prompt and persona style are embedded** for best results (no external files needed).
- **Usage example:**
  ```sh
  python extensions/character_persona_builder.py outputs/run-YYYYMMDD-HHMMSS --llm-config workflows/llm_tasks.json
  ```
- Outputs are written to `characters/<call_title or call_id>/<channel or conversation_speaker>/` with transcript, persona, and audio clips.

## Best Practices for Extension Authors
- Be robust to folder naming (handle run-specific prefixes, normalize for output).
- Log only anonymized, PII-free information.
- Support both batch and single-file workflows.
- Document your extension's purpose and usage.
- Follow privacy and traceability rules.

## General Extension Workflow
- Place your extension scripts in the `extensions/` directory.
- Each extension should inherit from `ExtensionBase` (see `extension_base.py`).
- Extensions are run manually or can be invoked automatically after pipeline completion.
- Extensions receive the root output directory as their argument and should only access finalized outputs.

## Example Extension Usage
```sh
python extensions/character_persona_builder.py outputs/run-YYYYMMDD-HHMMSS --llm-config workflows/llm_tasks.json
```

See the main project README for more details and extension authoring tips.

## How Extensions Work
- Place your extension scripts in the `extensions/` directory.
- Each extension should inherit from `ExtensionBase` (see `extension_base.py`).
- Extensions are run manually or can be invoked automatically after pipeline completion.
- Extensions receive the root output directory as their argument and should only access finalized outputs.

## Creating an Extension
1. **Inherit from `ExtensionBase`:**
   ```python
   from extension_base import ExtensionBase
   class MyExtension(ExtensionBase):
       def run(self):
           # Your logic here
           self.log("Running my extension!")
   ```
2. **Implement the `run()` method:**
   - Access outputs via `self.output_root`.
   - Use `self.manifest` to read the manifest if needed.
   - Log only anonymized, PII-free information.

3. **Run your extension:**
   ```sh
   python my_extension.py <output_root>
   ```

## Best Practices
- **Privacy:** Never access or log original filenames, paths, or PII. Only use anonymized, finalized data.
- **Traceability:** Use the manifest and output folder structure for all data lineage.
- **Idempotence:** Extensions should be safe to run multiple times.
- **Robustness:** Handle missing or partial data gracefully.

## Example Extensions
- `transcript_and_soundbite_cleanup.py`: Aggregates transcripts and cleans up obsolete soundbites folders.
- Analytics, visualizations, LLM-based summaries, and more can be added as new extensions.

## Contributing
- Document your extension's purpose and usage.
- Follow the privacy and traceability rules outlined above. 