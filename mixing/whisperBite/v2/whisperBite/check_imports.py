"""
Script to check for proper imports in WhisperBite package.
"""

import os
import ast
import sys
from typing import List, Set

def get_python_files(directory: str) -> List[str]:
    """Get all Python files in directory and subdirectories."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def check_imports(file_path: str) -> List[str]:
    """Check imports in a Python file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return [f"Syntax error in {file_path}: {str(e)}"]
    
    issues = []
    local_imports = set()
    
    # Get current module context
    current_dir = os.path.basename(os.path.dirname(file_path))
    
    whisperBite_modules = {
        'feature_extractor', 'normalizer', 'transcriber', 'diarizer',
        'word_splitter', 'vocal_separator', 'demucs_processor', 'post_processor'
    }
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                base_name = name.name.split('.')[0]
                if base_name in whisperBite_modules:
                    issues.append(f"Direct import of {name.name} in {file_path}")
                    local_imports.add(name.name)
                        
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base_module = node.module.split('.')[0]
                if base_module in whisperBite_modules:
                    # Only flag if it's not a relative import (level == 0)
                    if node.level == 0 and not node.module.startswith('whisperBite'):
                        issues.append(f"Direct import from {node.module} in {file_path}")
                        local_imports.add(base_module)
    
    if local_imports:
        issues.append(f"\nSuggested fixes for {file_path}:")
        for imp in local_imports:
            if current_dir in ('core', 'processors', 'utils'):
                # For files in main subdirectories, suggest relative imports
                issues.append(f"  Change: from {imp}")
                issues.append(f"  To:     from .{imp} import ...")
            else:
                # For files outside main subdirectories, suggest absolute imports
                issues.append(f"  Change: from {imp}")
                issues.append(f"  To:     from whisperBite.core.{imp} or appropriate module")
    
    return issues

def main():
    """Main function to check all Python files."""
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    python_files = get_python_files(base_dir)
    has_issues = False
    
    print(f"Checking Python files in {base_dir}\n")
    
    for file_path in python_files:
        issues = check_imports(file_path)
        if issues:
            has_issues = True
            print(f"\nIssues in {os.path.relpath(file_path, base_dir)}:")
            for issue in issues:
                print(f"  {issue}")
    
    if not has_issues:
        print("No import issues found!")
        return 0
    return 1

if __name__ == "__main__":
    sys.exit(main())