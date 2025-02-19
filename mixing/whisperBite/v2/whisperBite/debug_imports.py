"""
Debug version of import checker to understand why it's flagging correct imports.
"""

import os
import ast
import sys
from typing import List

def check_imports(file_path: str) -> List[str]:
    """Check imports in a Python file with debug output."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    print(f"\nDebug: Checking file {file_path}")
    print("Content:")
    print(content[:200] + "...") # Show first 200 chars
    
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return [f"Syntax error in {file_path}: {str(e)}"]
    
    issues = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            print(f"\nFound ImportFrom node:")
            print(f"  module: {node.module}")
            print(f"  names: {[n.name for n in node.names]}")
            print(f"  level: {node.level}")  # level > 0 indicates relative import
            
            if node.module:
                base_module = node.module.split('.')[0]
                print(f"  base_module: {base_module}")
                
                # Check if this is actually a relative import
                is_relative = node.level > 0
                print(f"  is_relative: {is_relative}")
                
                if not is_relative and base_module in {
                    'demucs_processor', 'post_processor'
                }:
                    issues.append(f"Direct import from {node.module}")
    
    return issues

def main():
    """Debug main function."""
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    target_file = os.path.join(base_dir, "whisperBite", "processors", "vocal_separator.py")
    if os.path.exists(target_file):
        issues = check_imports(target_file)
        if issues:
            print("\nIssues found:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("\nNo issues found!")
    else:
        print(f"Could not find {target_file}")

if __name__ == "__main__":
    main()
