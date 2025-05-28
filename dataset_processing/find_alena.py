import os
import re
from tqdm import tqdm
import glob
import sys

def find_prt_files_with_keyword(folder_path, keyword="Schillerova"):
    matching_files = []  # Store paths of matching files
    
    paths = glob.glob(os.path.join(folder_path, '**', '*.speakers'), recursive=True)
    if not paths:
        print(f"No .speakers files found in {folder_path}.")
        return matching_files

    for path in tqdm(paths):
        try:
            # Open the file in binary mode and read its content
            with open(path, 'r') as f:
                content = f.read()
                
                if re.search(keyword, content):
                    matching_files.append(path)
                    print(f"Match found in: {path}")

        except Exception as e:
            print(f"Error reading file {path}: {e}")
    
    return matching_files


if __name__ == "__main__":
    folder_to_search = "/mnt/matylda4/xluner01/ParCzech/parczech-3.0-asr-train-2021/"
    
    # Find all .prt files containing "AlenaSchillerova" and print their paths
    matching_files = find_prt_files_with_keyword(folder_to_search)
    
    if matching_files:
        print("\nMatching files:")
        for path in matching_files:
            print(path)
    else:
        print("\nNo matching files found.")
