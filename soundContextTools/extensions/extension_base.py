import os
import json
from pathlib import Path

class ExtensionBase:
    """
    Base class for all extension scripts. Provides standardized access to finalized outputs and manifest,
    and enforces privacy/logging rules.
    """
    def __init__(self, output_root):
        self.output_root = Path(output_root)
        self.manifest = self._load_manifest()

    def _load_manifest(self):
        manifest_path = self.output_root / 'manifest.json'
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def log(self, message):
        # Only log anonymized, PII-free information
        print(f"[EXTENSION LOG] {message}")

    def run(self):
        raise NotImplementedError("Extension must implement the run() method.")

# Example usage:
# class MyExtension(ExtensionBase):
#     def run(self):
#         # Your logic here
#         self.log("Running my extension!") 