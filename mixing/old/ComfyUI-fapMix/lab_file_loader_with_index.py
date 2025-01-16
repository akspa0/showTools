import os
from pathlib import Path

class LabFileLoaderWithIndexNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory_path": ("STRING", {"multiline": False, "default": ""}),
                "file_index": ("INT", {"default": 0, "min": 0}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("lab_file_content", "lab_file_name")
    FUNCTION = "load_lab_file_by_index"
    CATEGORY = "Custom Nodes/Text"

    def load_lab_file_by_index(self, directory_path: str, file_index: int) -> (str, str):
        # Validate the directory
        if not os.path.isdir(directory_path):
            print(f"Directory not found: {directory_path}")
            return ("", "")
        # Load .lab files
        lab_files = sorted(Path(directory_path).glob("*.lab"))

        if not lab_files:
            print("No .lab files found in the directory.")
            return ("", "")

        if file_index < 0 or file_index >= len(lab_files):
            print(f"File index {file_index} is out of range (0 to {len(lab_files)-1}).")
            return ("", "")

        # Load the .lab file at the specified index
        lab_file_path = lab_files[file_index]
        with open(lab_file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        print(f"Loaded .lab file: {lab_file_path}")
        # Sanitize the text if needed
        sanitized_text = text.lstrip(' ?').strip()

        return (sanitized_text, lab_file_path.name)

# Register the node
NODE_CLASS_MAPPINGS = {
    "Lab File Loader With Index": LabFileLoaderWithIndexNode
}
