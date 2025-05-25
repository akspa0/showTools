# Extension System for Audio Context Tool

## Overview
Extensions allow you to add post-processing, context generation, and derivative art/analytics to the finalized outputs of the pipeline. Extensions run after the main pipeline and operate only on anonymized, finalized data.

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