import os
import fnmatch
from typing import List

def find_files(directory: str, pattern: str, recursive: bool = True) -> List[str]:
    matched_files = []
    abs_directory = os.path.abspath(directory)
    
    for root, dirs, files in os.walk(abs_directory):
        for filename in files:
            if fnmatch.fnmatch(filename, pattern):
                matched_files.append(os.path.join(root, filename))
        
        if not recursive:
            break
            
    return matched_files